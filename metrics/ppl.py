"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import torch
import torch.nn as nn
from torchvision import models


def normalize(x, eps=1e-10):
    return x * torch.rsqrt(torch.sum(x**2, dim=1, keepdim=True) + eps)

def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)
class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        # vgg_pretrained_features.cuda()
        # print(vgg_pretrained_features)
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        # fullmodel = []
        # for x in range(16):
        #     fullmodel.append(vgg_pretrained_features[x])

        # print(fullmodel)

        # self.fullmodel = fullmodel
        # self.fullmodel.cuda()
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        
        h = self.slice1(X)
        h_relu1_1 = h
        h = self.slice2(h)
        h_relu2_1 = h
        h = self.slice3(h)
        h_relu3_1 = h
        h = self.slice4(h)
        h_relu4_1 = h
        h = self.slice5(h)
        h_relu5_1 = h  
#         vgg_outputs = namedtuple("VggOutputs", ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu_5_1'])
#         h_relu4_1,_,_ = norm_feat(h_relu4_1)
#         out = vgg_outputs(h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1, h_relu5_1)
        out =[h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1, h_relu5_1]
        # vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3'])
        # out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3)

        # fullout = []
        # fullout.cuda()
        # inter_data = X
        # print(inter_data)
        # for x in range(16):
        #     temp_output = self.fullmodel[x](inter_data)
        #     temp_output.cuda()
        #     fullout.append(temp_output)
        #     inter_data = temp_output

        

        return out 

class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        # vgg_pretrained_features.cuda()
        # print(vgg_pretrained_features)
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        # fullmodel = []
        # for x in range(16):
        #     fullmodel.append(vgg_pretrained_features[x])

        # print(fullmodel)

        # self.fullmodel = fullmodel
        # self.fullmodel.cuda()
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        
        h = self.slice1(X)
        h_relu1_1 = h
        h = self.slice2(h)
        h_relu2_1 = h
        h = self.slice3(h)
        h_relu3_1 = h
        h = self.slice4(h)
        h_relu4_1 = h
        h = self.slice5(h)
        h_relu5_1 = h  
#         vgg_outputs = namedtuple("VggOutputs", ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu_5_1'])
#         h_relu4_1,_,_ = norm_feat(h_relu4_1)
#         out = vgg_outputs(h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1, h_relu5_1)
        out =[h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1]
        # vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3'])
        # out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3)

        # fullout = []
        # fullout.cuda()
        # inter_data = X
        # print(inter_data)
        # for x in range(16):
        #     temp_output = self.fullmodel[x](inter_data)
        #     temp_output.cuda()
        #     fullout.append(temp_output)
        #     inter_data = temp_output

        

        return out 
    
class VGGLoss(nn.Module):
    def __init__(self, start_lay=0):
        super(VGGLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.weights = [1.0, 1.0, 1.0, 1.0]
        self.start_lay = start_lay

    def forward(self, x_vgg, x_rec_vgg):
        loss = 0
        for i in range(self.start_lay, len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_rec_vgg[i], x_vgg[i].detach())
        return loss


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = models.alexnet(pretrained=True).features
        self.channels = []
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                self.channels.append(layer.out_channels)

    def forward(self, x):
        fmaps = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                fmaps.append(x)
        return fmaps


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))

    def forward(self, x):
        return self.main(x)


class PPL(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = Vgg16().cuda()
#         self.vggloss = VGGLoss()
#         self.lpips_weights = nn.ModuleList()
#         for channels in self.alexnet.channels:
#             self.lpips_weights.append(Conv1x1(channels, 1))
#         self._load_lpips_weights()
#         # imagenet normalization for range [-1, 1]
#         self.mu = torch.tensor([-0.03, -0.088, -0.188]).view(1, 3, 1, 1).cuda()
#         self.sigma = torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1).cuda()

#     def _load_lpips_weights(self):
#         own_state_dict = self.state_dict()
#         if torch.cuda.is_available():
#             state_dict = torch.load('metrics/lpips_weights.ckpt')
#         else:
#             state_dict = torch.load('metrics/lpips_weights.ckpt',
#                                     map_location=torch.device('cpu'))
#         for name, param in state_dict.items():
#             if name in own_state_dict:
#                 own_state_dict[name].copy_(param)

    def forward(self, x, y):
        x = (x + 1) / 2
        y = (y +1 ) /2
        means=[0.485, 0.456, 0.406]
        stds=[0.229, 0.224, 0.225]
        for i in range(3):
            x[:,i,:,:] -= means[i]
            x[:,i,:,:] /= stds[i]
            
            y[:,i,:,:] -= means[i]
            y[:,i,:,:] /= stds[i]
            
            
        vgg_x = self.vgg(x)
        vgg_y = self.vgg(y)
        eps = 1e-4
        dist = 0
        for i in range(len(vgg_x)):
            v_x  = normalize_tensor(vgg_x[i])
            v_y = normalize_tensor(vgg_y[i])
            dist += torch.mean(torch.sum((v_x - v_y)**2,dim=0))
        
        dist = dist*((1/eps)**2)
#         x = (x - self.mu) / self.sigma
#         y = (y - self.mu) / self.sigma
#         x_fmaps = self.alexnet(x)
#         y_fmaps = self.alexnet(y)
#         lpips_value = 0
#         for x_fmap, y_fmap, conv1x1 in zip(x_fmaps, y_fmaps, self.lpips_weights):
#             x_fmap = normalize(x_fmap)
#             y_fmap = normalize(y_fmap)
#             lpips_value += torch.mean(conv1x1((x_fmap - y_fmap)**2))
        return dist


@torch.no_grad()
def calculate_ppl_given_images(x1,x2):
    # group_of_images = [torch.randn(N, C, H, W) for _ in range(10)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ppl = PPL().eval().to(device)
#     ppl_total = 0
#     for i in range(len(group1)):
    ppl_total = ppl(x1,x2)
#     ppl_total /= len(group1)
#     lpips_values = []
#     num_rand_outputs = len(group_of_images)

#     # calculate the average of pairwise distances among all random outputs
#     for i in range(num_rand_outputs-1):
#         for j in range(i+1, num_rand_outputs):
#             lpips_values.append(lpips(group_of_images[i], group_of_images[j]))
#     lpips_value = torch.mean(torch.stack(lpips_values, dim=0))
    return ppl_total.item()