import time
import torch.optim
from MYCNET import MYCNET
from time import *
import cv2
import torch
import numpy as np
import os
import scipy.io as scio
import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-arch', type=str, default='MYCNET')
    parser.add_argument('-root', type=str, default='./')
    parser.add_argument('-dataset', type=str, default='Pavia',
                        choices=['Botswana',  'Urban', 'Pavia'])
    parser.add_argument('--scale_ratio', type=float, default=4)
    parser.add_argument('--n_bands', type=int, default=0)
    parser.add_argument('--n_select_bands', type=int, default=5)
    parser.add_argument('--model_path', type=str,
                        default='./checkpoints/dataset_arch.pkl',
                        help='path for trained encoder')
    parser.add_argument('--n_epochs', type=int, default=10000,
                        help='end epoch for training')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=128)
    args = parser.parse_args()
    return args



def build_datasets(root, dataset, size, n_select_bands, scale_ratio):
    if dataset == 'Pavia':
        img = scio.loadmat(root + '/' + 'Pavia.mat')['pavia']*1.0
    elif dataset == 'Botswana':
        img = scio.loadmat(root + '/' + 'Botswana.mat')['Botswana']*1.0
    elif dataset == 'Urban':
        img = scio.loadmat(root + '/' + 'Urban.mat')['Y']
        img = np.reshape(img, (162, 307, 307))*1.0
        img = np.swapaxes(img, 0,2)

    print (img.shape)
    max = np.max(img)
    min = np.min(img)
    img = 255*((img - min) / (max - min + 0.0))

    w_edge = img.shape[0]//scale_ratio*scale_ratio-img.shape[0]
    h_edge = img.shape[1]//scale_ratio*scale_ratio-img.shape[1]
    w_edge = -1  if w_edge==0  else  w_edge
    h_edge = -1  if h_edge==0  else  h_edge
    img = img[:w_edge, :h_edge, :]

    width, height, n_bands = img.shape
    w_str = (width - size) // 2
    h_str = (height - size) // 2
    w_end = w_str + size
    h_end = h_str + size
    img_copy = img.copy()

    gap_bands = n_bands / (n_select_bands-1.0)
    test_ref = img_copy[w_str:w_end, h_str:h_end, :].copy()
    test_lr = cv2.GaussianBlur(test_ref, (5,5), 2)
    test_lr = cv2.resize(test_lr, (size//scale_ratio, size//scale_ratio))

    test_hr = test_ref[:,:,0][:,:,np.newaxis]
    for i in range(1, n_select_bands-1):
        test_hr = np.concatenate((test_hr, test_ref[:,:,int(gap_bands*i)][:,:,np.newaxis],), axis=2)
    test_hr = np.concatenate((test_hr, test_ref[:,:,n_bands-1][:,:,np.newaxis],), axis=2)

    img[w_str:w_end,h_str:h_end,:] = 0
    train_ref = img
    train_lr = cv2.GaussianBlur(train_ref, (5,5), 2)
    train_lr = cv2.resize(train_lr, (train_lr.shape[1]//scale_ratio, train_lr.shape[0]//scale_ratio))
    train_hr = train_ref[:,:,0][:,:,np.newaxis]
    for i in range(1, n_select_bands-1):
        train_hr = np.concatenate((train_hr, train_ref[:,:,int(gap_bands*i)][:,:,np.newaxis],), axis=2)
    train_hr = np.concatenate((train_hr, train_ref[:,:,n_bands-1][:,:,np.newaxis],), axis=2)


    train_ref = torch.from_numpy(train_ref).permute(2,0,1).unsqueeze(dim=0)
    train_lr = torch.from_numpy(train_lr).permute(2,0,1).unsqueeze(dim=0)
    train_hr = torch.from_numpy(train_hr).permute(2,0,1).unsqueeze(dim=0)
    test_ref = torch.from_numpy(test_ref).permute(2,0,1).unsqueeze(dim=0)
    test_lr = torch.from_numpy(test_lr).permute(2,0,1).unsqueeze(dim=0)
    test_hr = torch.from_numpy(test_hr).permute(2,0,1).unsqueeze(dim=0)

    return [train_ref, train_lr, train_hr], [test_ref, test_lr, test_hr]

def calc_ergas(img_tgt, img_fus):
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)

    rmse = np.mean((img_tgt-img_fus)**2, axis=1)
    rmse = rmse**0.5
    mean = np.mean(img_tgt, axis=1)

    ergas = np.mean((rmse/mean)**2)
    ergas = 100/4*ergas**0.5

    return ergas

def calc_psnr(img_tgt, img_fus):
    mse = np.mean((img_tgt-img_fus)**2)
    img_max = np.max(img_tgt)
    psnr = 10*np.log10(img_max**2/mse)

    return psnr

def calc_rmse(img_tgt, img_fus):
    rmse = np.sqrt(np.mean((img_tgt-img_fus)**2))

    return rmse

def calc_sam(img_tgt, img_fus):
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    img_tgt = img_tgt / np.max(img_tgt)
    img_fus = img_fus / np.max(img_fus)

    A = np.sqrt(np.sum(img_tgt**2, axis=0))
    B = np.sqrt(np.sum(img_fus**2, axis=0))
    AB = np.sum(img_tgt*img_fus, axis=0)

    sam = AB/(A*B)
    sam = np.arccos(sam)
    sam = np.mean(sam)*180/3.1415926535

    return sam





args = args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print (args)


def main():
    if args.dataset == 'Pavia':
      args.n_bands = 102
    elif args.dataset == 'Botswana':
      args.n_bands = 145
    elif args.dataset == 'Urban':
      args.n_bands = 162


    train_list, test_list = build_datasets(args.root, 
                                           args.dataset, 
                                           args.image_size, 
                                           args.n_select_bands, 
                                           args.scale_ratio)

    model = MYCNET(args.arch,
                 args.scale_ratio,
                 args.n_select_bands,
                 args.n_bands).cuda()

    model_path = args.model_path.replace('dataset', args.dataset) \
                                .replace('arch', args.arch) 
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path), strict=False)
        print ('Load the chekpoint of {}'.format(model_path))


    test_ref, test_lr, test_hr = test_list
    model.eval()

    ref = test_ref.float().detach()
    lr = test_lr.float().detach()
    hr = test_hr.float().detach()

    ref=ref.cuda()
    lr=lr.cuda()
    hr=hr.cuda()

    begin_time = time()
    out, _, _, _, _, _ = model(lr, hr)


    end_time = time()
    run_time = (end_time-begin_time)*1000

    print ('Dataset:   {}'.format(args.dataset))
    print ('Arch:   {}'.format(args.arch))


    ref = ref.detach().cpu().numpy()
    out = out.detach().cpu().numpy()
    
    psnr = calc_psnr(ref, out)
    rmse = calc_rmse(ref, out)
    ergas = calc_ergas(ref, out)
    sam = calc_sam(ref, out)
    print ('RMSE:   {:.4f};'.format(rmse))
    print ('PSNR:   {:.4f};'.format(psnr))
    print ('ERGAS:   {:.4f};'.format(ergas))
    print ('SAM:   {:.4f}.'.format(sam))

if __name__ == '__main__':
    main()
