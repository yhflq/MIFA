import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torch.nn import init
# from models.SelfAttention import ScaledDotProductAttention
# from models.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention
# from fvcore.nn import FlopCountAnalysis, parameter_count, parameter_count_table
from tqdm import tqdm
import time


# print('x_lr',x_lr.shape)   #([1, 145, 32, 32])
#  print('x_hr', x_hr.shape)  #([1, 5, 128, 128])
def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))


def flops_params_fps(model, input_shape1=(1, 145, 32, 32), input_shape2=(1, 5, 128, 128)):
    """count flops:G params:M fps:img/s
        input shape tensor[1, c, h, w]
    """
    total_time = []
    with torch.no_grad():
        model = model.cuda().eval()
        input1 = torch.randn(size=input_shape1, dtype=torch.float32).cuda()
        input2 = torch.randn(size=input_shape2, dtype=torch.float32).cuda()
        flops = FlopCountAnalysis(model, [input1, input2])
        params = parameter_count(model)

        for i in tqdm(range(100)):
            torch.cuda.synchronize()
            start = time.time()
            output = model(input1, input2)
            torch.cuda.synchronize()
            end = time.time()
            total_time.append(end - start)
        mean_time = np.mean(np.array(total_time))
        print(model.__class__.__name__)
        print('img/s:{:.2f}'.format(1 / mean_time))
        print('flops:{:.2f}G params:{:.2f}M'.format(flops.total() / 1e9, params[''] / 1e6))




class Spe_Spa_Attention(nn.Module):

    def __init__(self, channel=512):
        super(Spe_Spa_Attention, self).__init__()
        self.ch_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.ch_wq = nn.Conv2d(channel, 1, kernel_size=(1, 1))
        self.softmax_channel = nn.Softmax(1)
        self.softmax_spatial = nn.Softmax(-1)
        self.ch_wz = nn.Conv2d(channel // 2, channel, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(channel)
        self.sigmoid = nn.Sigmoid()
        self.sp_wv = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.sp_wq = nn.Conv2d(channel, channel // 2, kernel_size=(1, 1))
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        b, c, h, w = x.size()

        channel_wv = self.ch_wv(x)
        channel_wq = self.ch_wq(x)
        channel_wv = channel_wv.reshape(b, c // 2, -1)
        channel_wq = channel_wq.reshape(b, -1, 1)
        channel_wq = self.softmax_channel(channel_wq)
        channel_wz = torch.matmul(channel_wv, channel_wq).unsqueeze(-1)
        channel_weight = self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b, c, 1).permute(0, 2, 1))).permute(0, 2,1).reshape(b, c, 1, 1)
        channel_out = channel_weight * x
        spatial_wv = self.sp_wv(x)
        spatial_wq = self.sp_wq(x)
        spatial_wq = self.agp(spatial_wq)
        spatial_wv = spatial_wv.reshape(b, c // 2, -1)
        spatial_wq = spatial_wq.permute(0, 2, 3, 1).reshape(b, 1, c // 2)
        spatial_wq = self.softmax_spatial(spatial_wq)
        spatial_wz = torch.matmul(spatial_wq, spatial_wv)
        spatial_weight = self.sigmoid(spatial_wz.reshape(b, 1, h, w))
        spatial_out = spatial_weight * x
        out = spatial_out + channel_out
        return out

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale



class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CoTNetLayer(nn.Module):

    def __init__(self, dim=512, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            # 通过K*K的卷积提取上下文信息，视作输入X的静态上下文表达
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False),  # 1*1的卷积进行Value的编码
            nn.BatchNorm2d(dim)
        )

        factor = 4
        self.attention_embed = nn.Sequential(  # 通过连续两个1*1的卷积计算注意力矩阵
            nn.Conv2d(2 * dim, 2 * dim // factor, 1, bias=False),  # 输入concat后的特征矩阵 Channel = 2*C
            nn.BatchNorm2d(2 * dim // factor),
            nn.ReLU(),
            nn.Conv2d(2 * dim // factor, kernel_size * kernel_size * dim, 1, stride=1)  # out: H * W * (K*K*C)
        )

    def forward(self, x):
        bs, c, h, w = x.shape
        # print('x_cot',x.shape)
        k1 = self.key_embed(x)  # shape：bs,c,h,w  提取静态上下文信息得到key
        # print('k1',k1.shape)
        v = self.value_embed(x).view(bs, c, -1)  # shape：bs,c,h*w  得到value编码

        y = torch.cat([k1, x], dim=1)  # shape：bs,2c,h,w  Key与Query在channel维度上进行拼接进行拼接
        att = self.attention_embed(y)  # shape：bs,c*k*k,h,w  计算注意力矩阵
        att = att.reshape(bs, c, self.kernel_size * self.kernel_size, h, w)
        att = att.mean(2, keepdim=False).view(bs, c, -1)  # shape：bs,c,h*w  求平均降低维度
        k2 = F.softmax(att, dim=-1) * v  # 对每一个H*w进行softmax后
        k2 = k2.view(bs, c, h, w)

        return k1 + k2  # 注意力融合


class COT_Block(nn.Module):
    def __init__(self, c_h, o_h, ks=1):
        super(COT_Block, self).__init__()
        # self.conv1 = ConvBNReLU(c_h, o_h, ks=ks, bias=False)

        self.conv1 = nn.Conv2d(c_h, o_h, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(o_h)

        self.cot2 = CoTNetLayer(dim=o_h, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(o_h)
        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(o_h, c_h, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(c_h)

    def forward(self, x):
        # print('x',x.shape)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out2 = self.cot2(out)
        out2 = self.bn2(out2)
        out2 = self.relu(out2)

        out3 = self.conv3(out2)  # 1*1 Conv
        out3 = self.bn3(out3)

        out3 += residual
        out_F = self.relu(out3)

        return out_F

class imp_self_att(nn.Module):
    def __init__(self, channels):
        super(imp_self_att, self).__init__()
        self.conv_value = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.conv_query = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_key = nn.Conv2d(channels, channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        value = self.conv_value(y)
        value = value.view(value.size(0), value.size(1), -1)

        query = self.conv_query(x)
        key = self.conv_key(y)
        query = query.view(query.size(0), query.size(1), -1)
        key = key.view(key.size(0), key.size(1), -1)

        key_mean = key.mean(2).unsqueeze(2)
        query_mean = query.mean(2).unsqueeze(2)
        key -= key_mean
        query -= query_mean

        sim_map = torch.bmm(query.transpose(1, 2), key)
        sim_map = self.softmax(sim_map)
        out_sim = torch.bmm(sim_map, value.transpose(1, 2))
        out_sim = out_sim.transpose(1, 2)
        out_sim = out_sim.view(out_sim.size(0), out_sim.size(1), *x.size()[2:])
        out_sim = self.gamma * out_sim

        return out_sim


class atten2(nn.Module):
    def __init__(self, channels_trans, channels_cnn, channels_fuse, residual=False):
        super(atten2, self).__init__()
        self.channels_fuse = channels_fuse
        self.residual = residual
        self.conv_trans = nn.Conv2d(channels_trans, channels_fuse, kernel_size=1)
        self.conv_cnn = nn.Conv2d(channels_cnn, channels_fuse, kernel_size=1)
        self.conv_fuse = nn.Conv2d(2 * channels_fuse, 2 * channels_fuse, kernel_size=1)
        self.conv1 = nn.Conv2d(channels_fuse, channels_fuse, kernel_size=1)
        self.conv2 = nn.Conv2d(channels_fuse, channels_fuse, kernel_size=1)
        self.softmax = nn.Softmax(dim=0)
        if residual:
            self.conv = nn.Conv2d(channels_trans + channels_cnn, channels_fuse, kernel_size=1)

    def forward(self, x, y):
        if self.residual:
            residual = self.conv(torch.cat((x, y), dim=1))
        x = self.conv_trans(x)
        y = self.conv_cnn(y)
        x_ori, y_ori = x, y
        xy_fuse = self.conv_fuse(torch.cat((x, y), 1))
        xy_split = torch.split(xy_fuse, self.channels_fuse, dim=1)
        x = torch.sigmoid(self.conv1(xy_split[0]))
        y = torch.sigmoid(self.conv2(xy_split[1]))
        weights = self.softmax(torch.stack((x, y), 0))
        out = weights[0] * x_ori + weights[1] * y_ori
        if self.residual:
            out = out + residual

        return out


class MYCNET(nn.Module):
    def __init__(self,arch, scale_ratio, n_select_bands, n_bands):
        super(MYCNET, self).__init__()
        self.scale_ratio = scale_ratio
        self.n_bands = n_bands
        self.arch = arch
        self.n_select_bands = n_select_bands
        self.weight = nn.Parameter(torch.tensor([0.5]))
        if self.n_bands == 145:
            self.n_ch = 144
        if self.n_bands == 162:
            self.n_ch = 164
        if self.n_bands == 102:
            self.n_ch = 144

        self.lr_conv1 = nn.Sequential(
            nn.Conv2d(n_bands, self.n_ch, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        self.spa_att1 = spatial_attn_layer()

        self.conv_fus = nn.Sequential(
            nn.Conv2d(self.n_ch, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv_spat = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
        )
        self.conv_spec = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.hr_up = nn.Sequential(
            nn.Conv2d(self.n_select_bands, 96, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(96, n_bands, kernel_size=3, stride=1, padding=1),
        )

        self.hr_down = nn.Sequential(
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(n_bands, n_bands, kernel_size=3, stride=1, padding=1),
        )

        self.hr_lr_rong = nn.Sequential(
            nn.Conv2d(self.n_ch + self.n_select_bands, self.n_ch, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(self.n_ch, self.n_ch, kernel_size=3, stride=1, padding=1),
        )

        self.imp_self_att = imp_self_att(n_bands)

        self.COT_Block1 = COT_Block(self.n_ch, 36)
        self.spe_spa = Spe_Spa_Attention(channel=self.n_ch)

    def spatial_edge(self, x):
        edge1 = x[:, :, 0:x.size(2) - 1, :] - x[:, :, 1:x.size(2), :]
        edge2 = x[:, :, :, 0:x.size(3) - 1] - x[:, :, :, 1:x.size(3)]
        return edge1, edge2

    def spectral_edge(self, x):
        edge = x[:, 0:x.size(1) - 1, :, :] - x[:, 1:x.size(1), :, :]
        return edge

    def forward(self, x_lr, x_hr):

        x_lr_1 = x_lr
        x_lr_cs = F.interpolate(x_lr_1, scale_factor=2, mode='bilinear')
        x_lr_cs1 = F.interpolate(x_lr_cs, scale_factor=2, mode='bilinear')
        x_lr = self.lr_conv1(x_lr)
        x_lr1 = x_lr
        x_hr1 = self.spa_att1(x_hr)
        x_hr1_cs1 = self.hr_up(x_hr1)
        x_hr1_cs2 = self.hr_down(x_hr1_cs1)
        x333 = self.imp_self_att(x_hr1_cs2, x_lr_cs)
        x333 = F.interpolate(x333, scale_factor=2, mode='bilinear')
        x = self.hr_lr_rong(
            torch.cat((x_hr1, F.interpolate(x_lr1, scale_factor=self.scale_ratio, mode='bilinear')), dim=1))

        x_cs2 = self.spe_spa(x)
        x_cot = self.COT_Block1(x_cs2)
        x = self.conv_fus(x_cot) + x333
        x_spat = x + self.conv_spat(x) + x_hr1_cs1
        spat_edge1, spat_edge2 = self.spatial_edge(x_spat)
        x_spec = x_spat + self.conv_spec(x_spat) + x_lr_cs1
        spec_edge = self.spectral_edge(x_spec)
        x = x_spec

        return x, x_spat, x_spec, spat_edge1, spat_edge2, spec_edge


