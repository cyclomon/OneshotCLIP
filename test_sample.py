import argparse
import math
import random
import os
import sys
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from copy import deepcopy
import numpy

from argparse import Namespace
from PIL import Image

from model import Generator

def test(args, g_ema,g_source,device):

    imsave_path = os.path.join('samples', args.exp)
    model_path = os.path.join('checkpoints', args.exp)

    if not os.path.exists(imsave_path):
        os.makedirs(imsave_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with torch.set_grad_enabled(False):
        g_ema.eval()
        g_source.eval()
        
        sample_z = torch.randn(args.n_sample,args.latent,device=device)
        sample_many = torch.randn(10000,args.latent,device=device)
        w_many = g_source.style(sample_many)
        w_mean = torch.mean(w_many,dim=0,keepdim=True)
        sample_w = g_ema.style(sample_z)
        sample_w = torch.lerp(sample_w,w_mean,0.3).unsqueeze(1)
        sample_w = sample_w.repeat(1,18,1)
        w_mean = w_mean.unsqueeze(1).repeat(args.n_sample,18,1)
        
        if args.no_mix is False:
            input_lat = w_mean
            input_lat[:,:args.mix_idx,:] = sample_w[:,:args.mix_idx,:]
            sample, _ = g_ema([input_lat],input_is_latent=True)
        else:
            sample, _ = g_ema([sample_w],input_is_latent=True)
            
        sample_src, _ = g_source([sample_w],input_is_latent=True)

        utils.save_image(
            sample,
            f"%s/test.png" % (imsave_path),
            nrow=sample.size(0),
            normalize=True,
            range=(-1, 1),
        )
        utils.save_image(
            sample_src,
            f"%s/test_src.png" % (imsave_path),
            nrow=sample_src.size(0),
            normalize=True,
            range=(-1, 1),
        )



if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_sample", type=int, default=10)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--ckpt_source", type=str, default=None)
    parser.add_argument("--exp", type=str, default=None, required=True)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--mix_idx", type=int, default=11)
    parser.add_argument("--no_mix", action="store_true")
    
    args = parser.parse_args()

    torch.manual_seed(2)
    random.seed(2)

    n_gpu = 1
    args.distributed = n_gpu > 1

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_source = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    g_source.eval()
    
    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        ckpt_source = torch.load(args.ckpt_source, map_location=lambda storage, loc: storage)
        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
        except ValueError:
            pass
        g_ema.load_state_dict(ckpt["g_ema"], strict=False)
        g_source.load_state_dict(ckpt_source["g_ema"], strict=False)
    if args.distributed:
        g_ema = nn.parallel.DataParallel(g_ema)
        g_source.load_state_dict(ckpt_source["g_ema"], strict=False)
        
    test(args, g_ema,g_source, device)