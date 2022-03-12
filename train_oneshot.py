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
from metrics.lpips import LPIPS

from model import Generator, Extra
from model import Patch_Discriminator as Discriminator
from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from losses import PatchLoss,ConstLoss
import clip


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

def clip_normalize(image,device):
    image = 0.5*image+0.5
    image = F.interpolate(image,size=224,mode='bicubic')
    mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

def requires_grad(model, flag=True):
    for name, p in model.named_parameters():
        p.requires_grad = flag

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch
            
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(
        grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def rand_bbox(size, res,prop=None):
    W = size
    H = size
    cut_w = res
    cut_h = res
    if prop is not None:
        res = np.random.rand()*(prop[1]-prop[0])
        cut_w = int(res*W)
        cut_h = int(res*H)
    tx = np.random.randint(0,W-cut_w)
    ty = np.random.randint(0,H-cut_h)
    bbx1 = tx
    bby1 = ty
    return bbx1, bby1

def rand_sampling_mult(sizes,crop_size,num_crops,content_image,out_image,prop=None):
    bbxl=[]
    bbyl=[]
    crop_image = []
    tar_image = []

    for cc in range(num_crops):
        bbx1, bby1 = rand_bbox(sizes, crop_size,prop)
        crop_image.append(content_image[:,:,bby1:bby1+crop_size,bbx1:bbx1+crop_size])
        tar_image.append(out_image[:,:,bby1:bby1+crop_size,bbx1:bbx1+crop_size])
    crop_image = torch.cat(crop_image,dim=0)
    tar_image = torch.cat(tar_image,dim=0)
    return crop_image,tar_image


def train(args, loader,loader_or, generator, discriminator, extra, g_optim, d_optim, e_optim, g_ema, device, g_source, clip_model):
    loader = sample_data(loader)

    imsave_path = os.path.join('samples', args.exp)
    model_path = os.path.join('checkpoints', args.exp)
    
    if not os.path.exists(imsave_path):
        os.makedirs(imsave_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    pbar = range(args.iter)

    
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter,
                    dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}


    g_module = generator
    d_module = discriminator
    g_ema_module = g_ema

    accum = 0.5 ** (32 / (10 * 1000))


    lowp, highp = 0, args.highp


    requires_grad(g_source, False)

    sample_z = torch.randn(args.n_sample,args.latent,device=device)
    sample_many = torch.randn(10000,args.latent,device=device)
    w_many = g_source.style(sample_many)
    w_mean = torch.mean(w_many,dim=0,keepdim=True).clone().detach()
    w_mean.requires_grad=True

    w_orig = w_mean.clone().detach()
    
    ref_w = w_mean
    f_optim = optim.Adam(
        [ref_w],
        lr=args.f_lr,
        betas=(0.9, 0.999),
    )
    
    del sample_many
    
    augment = transforms.Compose([
    transforms.RandomAffine(degrees=15, translate=[0.1,0.1], shear=5),
    transforms.RandomPerspective(distortion_scale=0.7, p=1.0),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomErasing(scale=(.1, .4), ratio=(.3, 1/.3), p=0.7),
    transforms.Resize([224,224])
])
    
    calc_const = ConstLoss(args)
    calc_patch = PatchLoss(args)
    calc_lpips = LPIPS().cuda()
    sim = nn.CosineSimilarity()
    
    loader_or = sample_data(loader_or)
    
    real_img_or = next(loader_or)
    real_img_or = real_img_or.to(device)
    if args.cars:
        real_img_or[:,:,:64,:] = -1
        real_img_or[:,:,448:,:] = -1
    if not args.skip_init:
        for ii in range(args.init_iter):
            fake_img, _ = g_source([ref_w],input_is_latent=True)
            fake_aug = augment(fake_img.repeat(8,1,1,1))

            fake_features = clip_model.encode_image(clip_normalize(fake_aug,device))
            fake_features /= fake_features.clone().norm(dim=-1, keepdim=True)

            real_features = clip_model.encode_image(clip_normalize(real_img_or,device))
            real_features /= real_features.clone().norm(dim=-1, keepdim=True)

            clip_loss = (1.0-sim(real_features,fake_features)).mean()

            l2_loss= torch.mean((ref_w-w_orig)**2)
            
            fake_img = F.interpolate(fake_img,size=real_img_or.size(2))
            rec_loss = calc_lpips(fake_img,real_img_or) + torch.mean((fake_img-real_img_or)**2) 
            g_loss = args.lambda_optclip*clip_loss+ args.lambda_optl2*l2_loss + args.lambda_optrec*rec_loss 

            loss_dict["clip"] = clip_loss
            loss_dict["l2"] = l2_loss
            loss_dict["rec"] = rec_loss
            
            f_optim.zero_grad()
            g_loss.backward()
            f_optim.step()
            if get_rank() == 0:
                if ii % args.img_freq == 0:
                    with torch.set_grad_enabled(False):

                        sample, _ = g_source([ref_w],input_is_latent=True)
                        sample = F.interpolate(sample,size=real_img_or.size(2))
                        test_out = torch.cat([sample,real_img_or],dim=0)
                        utils.save_image(
                            test_out,
                            f"%s/ref.png" % (imsave_path),
                            nrow=int(args.n_sample ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )
                        torch.save(
                            {
                                'init_w' : ref_w

                            },
                            f"%s/init_w.pt" % (model_path),)
    if args.skip_init:
        ckpt_init = torch.load("%s/init_w.pt"% (model_path))
        ref_w = ckpt_init["init_w"]
        
        
    loader = sample_data(loader)
    for idx in pbar:
        i = idx + args.start_iter
        which = i % args.ref_freq 

        if i > args.iter:
            print("Done!")
            break
        if which >0:
            real_img = next(loader)
            real_img = real_img.to(device)
        else:
            real_img = next(loader_or)
            real_img = real_img.to(device)
            
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        requires_grad(extra, True)

        if which > 0:
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        else:
            noise = ref_w
            
        if which >0:
            fake_img, _ = generator(noise)
        else:
            fake_img,_ = generator([noise],input_is_latent=True)

        fake_pred, _ = discriminator(
            fake_img, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp))
        real_pred, _ = discriminator(
            real_img, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp), real=True)

        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        extra.zero_grad()
        d_loss.backward()
        d_optim.step()
        e_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred, _ = discriminator(
                real_img, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp))
            real_pred = real_pred.view(real_img.size(0), -1)
            real_pred = real_pred.mean(dim=1).unsqueeze(1)

            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            extra.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every +
             0 * real_pred[0]).backward()

            d_optim.step()
            e_optim.step()
            
        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(generator.style,False)
        requires_grad(generator.to_rgb1,False)
        requires_grad(generator.to_rgbs,False)
        requires_grad(discriminator, False)
        requires_grad(extra, False)
        
        if which > 0:
            noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        else:
            noise = ref_w
        
        if which>0:
            fake_img, _ = generator(noise)
            src_img, _ = g_source(noise)
        else:
            fake_img, _ = generator([noise],input_is_latent=True)

        fake_pred, _ = discriminator(
            fake_img, extra=extra, flag=which, p_ind=np.random.randint(lowp, highp))
        g_loss = g_nonsaturating_loss(fake_pred)
        
        if which>0:
            src_feat_single = clip_model.encode_image(clip_normalize(src_img,device))
            src_feat_single /= src_feat_single.clone().norm(dim=-1, keepdim=True)

            fake_feat_single = clip_model.encode_image(clip_normalize(fake_img,device))
            fake_feat_single /= fake_feat_single.clone().norm(dim=-1, keepdim=True)


            src_img_crop,fake_img_crop = rand_sampling_mult(args.size,args.crop_size,args.num_crop,src_img,fake_img) 
            
            fake_patch = clip_model.encode_image(clip_normalize(fake_img_crop,device))
            fake_patch /= fake_patch.clone().norm(dim=-1,keepdim=True)
            
            src_patch = clip_model.encode_image(clip_normalize(src_img_crop,device))
            src_patch /= src_patch.clone().norm(dim=-1,keepdim=True)

            p_loss = args.lambda_patch*calc_patch(fake_patch,src_patch).mean()
            c_loss = args.lambda_const*calc_const(fake_feat_single,src_feat_single)
            loss_dict["c"] = c_loss
            loss_dict["p"] = p_loss
            g_loss += c_loss
            g_loss += p_loss
        else:
            g_loss += calc_lpips(fake_img,real_img) + torch.mean((fake_img-real_img)**2)
        
        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        del  g_loss, d_loss, fake_img, fake_pred, real_img, real_pred

        accumulate(g_ema_module, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)
        
        if which >0:
            c_loss_val = loss_reduced["c"].mean().item()
            p_loss_val = loss_reduced["p"].mean().item()
        else:
            c_loss_val = 0
            p_loss_val = 0
        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f};  c: {c_loss_val:.4f}; p: {p_loss_val:.4f}"

                )
            )

            if i % args.img_freq == 0:
                with torch.set_grad_enabled(False):
                    g_ema.eval()
                    sample, _ = g_ema([sample_z.data])
                    sample_ref, _ = g_ema([ref_w],input_is_latent=True)
                    sample = torch.cat([sample,sample_ref],dim=0)
                    utils.save_image(
                        sample,
                        f"%s/{str(i).zfill(6)}.png" % (imsave_path),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )
                    del sample

            if (i % args.save_freq == 0) and (i > 0):
                torch.save(
                    {
                        "g_ema": g_ema.state_dict(),
                    },
                    f"%s/{str(i).zfill(6)}.pt" % (model_path),
                )


if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--iter", type=int, default=2001)
    parser.add_argument("--save_freq", type=int, default=1000)
    parser.add_argument("--img_freq", type=int, default=100)
    parser.add_argument("--highp", type=int, default=1)
    parser.add_argument("--ref_freq", type=int, default=4)
    parser.add_argument("--feat_ind", type=int, default=3)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--n_sample", type=int, default=4)
    parser.add_argument("--size", type=int, default=1024)

    parser.add_argument("--r1", type=float, default=10)


    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--mixing", type=float, default=0.9)

    parser.add_argument("--ckpt", type=str, default=None)

    parser.add_argument("--exp", type=str, default=None, required=True)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--f_lr", type=float, default=0.01)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--skip_init",action='store_true')
    parser.add_argument("--init_iter", type=int, default=1001)
    parser.add_argument("--lambda_optclip", type=float, default=1)
    parser.add_argument("--lambda_optl2", type=float, default=0.01)
    parser.add_argument("--lambda_optrec", type=float, default=1)
    parser.add_argument("--lambda_patch", type=float, default=1)
    parser.add_argument("--lambda_const", type=float, default=10)
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--num_crop", type=int, default=16)
    parser.add_argument("--cars", action="store_true")
    parser.add_argument("--nce_allbatch", action="store_true")
    parser.add_argument("--tau", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(1)
    random.seed(1)

    n_gpu = 1
    args.distributed = n_gpu > 1

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_source = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    extra = Extra().to(device)
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    e_optim = optim.Adam(
        extra.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    
    if args.ckpt is not None:
        print("load model:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        ckpt_source = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"], strict=False)
        g_source.load_state_dict(ckpt_source["g"], strict=False)
        g_ema.load_state_dict(ckpt["g_ema"], strict=False)

        discriminator.load_state_dict(ckpt["d"])


        if 'g_optim' in ckpt.keys():
            g_optim.load_state_dict(ckpt["g_optim"])
        if 'd_optim' in ckpt.keys():
            d_optim.load_state_dict(ckpt["d_optim"])

    if args.distributed:
        geneator = nn.parallel.DataParallel(generator)
        g_ema = nn.parallel.DataParallel(g_ema)
        g_source = nn.parallel.DataParallel(g_source)

        discriminator = nn.parallel.DataParallel(discriminator)
        extra = nn.parallel.DataParallel(extra)

    transform = transforms.Compose(
        [
            transforms.Resize([args.size,args.size]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    
    transform_or = transforms.Compose(
        [
            transforms.Resize([args.size,args.size]),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = MultiResolutionDataset(args.data_path, transform, args.size)
    dataset_or = MultiResolutionDataset(args.data_path, transform_or, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=False),
        drop_last=True,
    )
    loader_or = data.DataLoader(
        dataset_or,
        batch_size=1,
        sampler=data_sampler(dataset_or, shuffle=True, distributed=False),
        drop_last=True,
    )

    train(args, loader,loader_or, generator, discriminator, extra, g_optim,
          d_optim, e_optim, g_ema, device, g_source,clip_model)