# import libraries
import argparse
import os
import numpy as np

import torch
from torch import optim
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_value_
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from scipy.ndimage import zoom
from glob import glob
import nibabel as nib
from tqdm import tqdm
import cv2
import torch.nn.functional as F

from data_loader_10fold import MRIDataset

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Booloon value expected')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.00005, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu')
parser.add_argument('--cuda-idx', default='0', type=str,
                    help='cuda index')
parser.add_argument('--adaptive-lr', default=True, type=str2bool,
                    help='use adaptive learning rate or not')
parser.add_argument('--data-dropout', default = False, type = str2bool,
                    help = 'Shall we use data_dropout, a.k.a., train with randomed subset of the training data each epoch.')
parser.add_argument('--data-dropout-remaining-size', default = 100, type = int,
                    help = 'How many scans per mini batch?')
parser.add_argument('--input-T1', default=False, type=str2bool,
                    help='Do we have T1 as input? If yes it''s always the first input')
parser.add_argument('--input-DeepC', default=False, type=str2bool,
                    help='Do we have DeepC as input? If yes and if we have T1 it''s the second input. If yes and we do not have T1 it''s the first input.')
parser.add_argument('--DeepC-isotropic', default=False, type=str2bool,
                    help='Shall we use isotropic DeepC or use DeepC in their original voxel spacing?')
parser.add_argument('--DeepC-isotropic-crop', default=False, type=str2bool,
                    help = 'Shall we crop the isotropic DeepC? Only valid if DeepC-isotropic is True.')
parser.add_argument('--T1-normalization-method', default = 'max', type = str,
                    help = 'How to normalize the input T1 scans?')
parser.add_argument('--DeepC-normalization-method', default = 'NA', type = str,
                    help = 'How to normalize the input DeepC scans?')
parser.add_argument('--double-vgg', default=True, type=str2bool,
                    help='Use two vgg encoder or use two channels. Only relevant when having two inputs.')
parser.add_argument('--double-vgg-share-param', default = True, type = str2bool,
                    help = 'Do we want the double VGG encoding layers to share parameters?')
parser.add_argument('--save-prediction-numpy-dir', type = str)
parser.add_argument('--load-dir', help='The directory used to save the trained models (so that we can load them)', type=str)
parser.add_argument('--which-to-load', help='Which model to load', default='best', type=str)
parser.add_argument('--channel', help='produce activation for which channel', default=None, type=str)
parser.add_argument('--val-folder', default = 'None', type = str, help = 'which folder is validation dataset?')

def main():
    args = parser.parse_args()


    print('Are we using T1 as input? : ', args.input_T1)
    print('Are we using DeepC as input? : ', args.input_DeepC)
    print('Are we using isotropic DeepC instead of DeepC in CUres (if we use DeepC)? : ', args.DeepC_isotropic)
    print('Are we cropping the isotropic DeepC (if we use isotropic DeepC)? : ', args.DeepC_isotropic_crop)
    if args.DeepC_isotropic_crop and not args.DeepC_isotropic:
        print('Not using isotropic DeepC. Argument DeepC_isotropic_crop is meaning less. Setting it to False.')
        args.DeepC_isotropic_crop = False
    print('Are we using double VGG instead of double channel, in case we have 2 inputs? : ', args.double_vgg)
    if args.double_vgg and (not args.input_T1 or not args.input_DeepC):
        print('Not having both T1 and DeepC as input. Argument double_vgg is meaningless. Setting it to False.')
        args.double_vgg = False
    print('For double VGG, do we share the encoding layer parameters? : ', args.double_vgg_share_param)
    if args.double_vgg_share_param and not args.double_vgg:
        print('double_vgg_share_param = True and double_vgg = False. Incompatible. Setting double_vgg_share_param to False.')
        args.double_vgg_share_param = False

    print('We will be loading from this directory: ', args.load_dir)
    print('Which channel do we use for CAM? : ', args.channel)

    load_dir = args.load_dir
    which_to_load = args.which_to_load
    cuda_idx = args.cuda_idx
    input_T1 = args.input_T1
    input_DeepC = args.input_DeepC
    DeepC_isotropic = args.DeepC_isotropic
    DeepC_isotropic_crop = args.DeepC_isotropic_crop # default
    double_vgg = args.double_vgg
    batch_size = args.batch_size
    T1_normalization_method = args.T1_normalization_method
    DeepC_normalization_method = args.DeepC_normalization_method


    # load data   
    TestDataDir = '/media/sail/HDD10T/DeepC_SCZ-Score/10Fold_Dataset/'

    # data_fold_list=['fold1','fold11']
    data_fold_list=['fold3']
    val_index=args.val_folder

    if val_index in data_fold_list:
        data_fold_list.remove(val_index)
    
    train_list=data_fold_list
    #train_list=val_list
    print("=> training folders: ")
    print(*train_list)

    # TestDataDir = './dataset/test/'
    Test_MRIDataset = MRIDataset(DataDir = TestDataDir, mode = 'train', input_T1 = input_T1, input_DeepC = input_DeepC, double_vgg = double_vgg,
                            DeepC_isotropic = DeepC_isotropic, DeepC_isotropic_crop = DeepC_isotropic_crop, transform=transforms.Compose([ToTensor()]),
                     T1_normalization_method = T1_normalization_method, DeepC_normalization_method = DeepC_normalization_method,fold_list=train_list)
    Test_dataloader = DataLoader(Test_MRIDataset, batch_size = batch_size,
                               shuffle=False, num_workers=4)

    # Load the saved model.
    checkpoint_file = load_dir + '/checkpoint_%s.tar' % which_to_load
    # print('Hi I am here!!!!!!!!!!!!')
    # print('checkpoint path: ',checkpoint_file)
    # print()
    # print()
    # os.exit(0)
    device = torch.device(f"cuda:%s" % cuda_idx if (torch.cuda.is_available()) else "cpu")
        
    model = vgg11_bn(input_T1 = input_T1, input_DeepC = input_DeepC, \
            double_vgg = double_vgg, double_vgg_share_param = False, \
            DeepC_isotropic = DeepC_isotropic, DeepC_isotropic_crop = DeepC_isotropic_crop, channel = args.channel)
    model.to(device)

    if os.path.isfile(checkpoint_file):
        print("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        best_acc1 = checkpoint['best_acc1']

        # update the state dict from multigpu-dic to single gpu dic
        new_state_dict = OrderedDict()

        for k, v in checkpoint['state_dict'].items():
            if 'module' in k:
                k = k.replace('module.', '')
            if 'features' in k:
                k = k.replace('features', 'feature_extractor')
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_file))



    temp=[]
    for i in train_list:
        sub_fold1=glob(TestDataDir+i+'/CU_WB_affine_ACBV_iso1mm/*/*COBRE*.nii.gz')
        sub_fold2=glob(TestDataDir+i+'/CU_WB_affine_ACBV_iso1mm/*/*BrainGlu*.nii.gz')
        sub_fold3=glob(TestDataDir+i+'/CU_WB_affine_ACBV_iso1mm/*/*NMorph*.nii.gz')
        sub_fold=sub_fold1+sub_fold2+sub_fold3
        temp=temp+sub_fold

    test_DeepC = sorted(temp)
    print(f'ACBV path: {TestDataDir} / {train_list} CU_WB_affine_ACBV_iso1mm')
    print(f'Load ACBV. Total ACBV test number is: ' + str(len(test_DeepC)))



    temp=[]            
    for i in train_list:
        sub_fold1=glob(TestDataDir+str(i)+'/MNI152_affine_WB_iso1mm/*/*COBRE*.nii.gz')
        sub_fold2=glob(TestDataDir+str(i)+'/MNI152_affine_WB_iso1mm/*/*BrainGlu*.nii.gz')
        sub_fold3=glob(TestDataDir+str(i)+'/MNI152_affine_WB_iso1mm/*/*NMorph*.nii.gz')
        sub_fold=sub_fold1+sub_fold2+sub_fold3
        temp.extend(sub_fold)
    
    test_T1s = sorted(temp)
    print(f'T1 path: {TestDataDir} / {train_list} MNI152_affine_WB_iso1mm')
    print(f'Load T1. Total T1 test number is: ' + str(len(test_T1s)))


    folder = load_dir.split('/')[-1]

    if args.channel == 'DeepC':
        print('deriving DeepC activation...')

        model.eval()

        activation_path = './result/activation/' + folder

        for i, (data) in enumerate(tqdm(Test_dataloader)):
            # assign variables
            input_data_T1, input_data_DeepC = None, None
            if input_T1:
                if args.double_vgg:
                    input_data_T1 = data['T1'].unsqueeze(1)
                else:
                    input_data_T1 = torch.zeros(data['T1'].shape).unsqueeze(1)
            if input_DeepC:
                input_data_DeepC = data['DeepC'].unsqueeze(1)
            target = data['label']
            
            # to GPU
            if input_T1:
                input_data_T1 = input_data_T1.to(device)
            if input_DeepC:
                input_data_DeepC = input_data_DeepC.to(device)
            target = target.to(device)

            pred = model(input_data_T1, input_data_DeepC)

            pred[:, 1].backward()

            gradients = model.get_activation_gradient()

            pooled_gradients = torch.abs(torch.mean(gradients, dim = [0, 2, 3, 4]))

            activations = model.get_activations(input_data_T1, input_data_DeepC).detach()

            for idx in range(128):
                activations[:, idx, :, :, :] *= pooled_gradients[idx]

            heatmap = torch.mean(activations, 1).squeeze().cpu()

            heatmap = np.maximum(heatmap, 0)

            #print(torch.max(heatmap))
            #heatmap = nn.ReLU(heatmap)

            #all_pos = heatmap >= 0

            #if False in all_pos:

            #    print('------Negative numbers exist !')


            heatmap /= torch.max(heatmap)
            
            current_DeepC = nib.load(test_DeepC[i])

            # if args.double_vgg:
            #     resampled_heatmap = zoom(heatmap, tuple(np.array([200, 200, 200]) / np.array(heatmap.shape)))
            # elif args.DeepC_isotropic_crop:
            #     resampled_heatmap = zoom(heatmap, tuple(np.array([200, 200, 200]) / np.array(heatmap.shape)))
            # else:
            #     resampled_heatmap = zoom(heatmap, tuple(np.array([200, 200, 200]) / np.array(heatmap.shape)))

            resampled_heatmap = zoom(heatmap, tuple(np.array([192, 192, 192]) / np.array(heatmap.shape))) #cubic spline
            #resampled_heatmap = zoom(heatmap, tuple(np.array([200, 200, 200]) / np.array(heatmap.shape)),order=0) # NN
            #resampled_heatmap = zoom(heatmap, tuple(np.array([200, 200, 200]) / np.array(heatmap.shape)),order=1) # bilinear

            resized_heatmap = center_crop_or_pad(resampled_heatmap, current_DeepC.shape)
            
            new_heatmap_nii = nib.Nifti1Image(resized_heatmap, current_DeepC.affine, current_DeepC.header)
            
            os.makedirs(activation_path + '/DeepC/', exist_ok = True)

            nib.save(new_heatmap_nii, activation_path + '/DeepC/' + test_DeepC[i].split('/')[-1][:-7] + '_activation.nii.gz')

    elif args.channel == 'T1':
        model.eval()

        print('deriving T1 activation...')

        activation_path = './result/activation/' + folder

        for i, (data) in enumerate(tqdm(Test_dataloader)):
            # assign variables
            input_data_T1, input_data_DeepC = None, None
            if input_T1:
                input_data_T1 = data['T1'].unsqueeze(1)
            if input_DeepC:
                if args.double_vgg:
                    input_data_DeepC = data['DeepC'].unsqueeze(1)
                else:
                    input_data_DeepC = torch.zeros(data['DeepC'].shape).unsqueeze(1)
            target = data['label']
            
            # to GPU
            if input_T1:
                input_data_T1 = input_data_T1.to(device)
            if input_DeepC:
                input_data_DeepC = input_data_DeepC.to(device)
            target = target.to(device)

            pred = model(input_data_T1, input_data_DeepC)

            pred[:, 1].backward()

            gradients = model.get_activation_gradient()

            pooled_gradients = torch.abs(torch.mean(gradients, dim = [0, 2, 3, 4]))

            activations = model.get_activations(input_data_T1, input_data_DeepC).detach()

            for idx in range(128):
                activations[:, idx, :, :, :] *= pooled_gradients[idx]

            heatmap = torch.mean(activations, 1).squeeze().cpu()

            heatmap = np.maximum(heatmap, 0)

            #print(torch.max(heatmap))
            #heatmap = nn.ReLU(heatmap)

            #all_pos = heatmap >= 0

            #if False in all_pos:

            #    print('------Negative numbers exist !')


            heatmap /= torch.max(heatmap)
            
            current_T1 = nib.load(test_T1s[i])

            # if args.double_vgg:
            #     resampled_heatmap = zoom(heatmap, tuple(np.array([200, 200, 200]) / np.array(heatmap.shape)))
            # elif args.DeepC_isotropic_crop:
            #     resampled_heatmap = zoom(heatmap, tuple(np.array([200, 200, 200]) / np.array(heatmap.shape)))
            # else:
            #     resampled_heatmap = zoom(heatmap, tuple(np.array([200, 200, 200]) / np.array(heatmap.shape)))



            resampled_heatmap = zoom(heatmap, tuple(np.array([192, 192, 192]) / np.array(heatmap.shape))) #cubic spline
            #resampled_heatmap = zoom(heatmap, tuple(np.array([200, 200, 200]) / np.array(heatmap.shape)),order=0) # NN
            #resampled_heatmap = zoom(heatmap, tuple(np.array([200, 200, 200]) / np.array(heatmap.shape)),order=1) # bilinear


            resized_heatmap = center_crop_or_pad(resampled_heatmap, current_T1.shape)

            #print(resized_heatmap.shape)
            
            new_heatmap_nii = nib.Nifti1Image(resized_heatmap, current_T1.affine, current_T1.header)
            
            os.makedirs(activation_path + '/T1/', exist_ok = True)

            nib.save(new_heatmap_nii, activation_path + '/T1/' + test_T1s[i].split('/')[-1][:-7] + '_activation.nii.gz')



'''
all the helper functions
Modified from https://githubnew_heatmap_nii.com/pytorch/vision.git
'''
import math

import torch
import torch.nn as nn
import torch.nn.init as init

se_ratio = 16

class ToTensor(object):
    def __call__(self, sample):
        torch_sample = {}
        for key, value in sample.items():
            if key == 'label':
                torch_sample[key] = torch.from_numpy(np.array(value))
            else:
                torch_sample[key] = torch.from_numpy(value)

        return torch_sample
    
class VGG(nn.Module): 
    '''
    VGG model 
    '''
    def __init__(self, feature_extractor, input_T1, input_DeepC, double_vgg, double_vgg_share_param, DeepC_isotropic, DeepC_isotropic_crop, channel = None):
        super(VGG, self).__init__()
        self.input_T1 = input_T1
        self.input_DeepC = input_DeepC

        self.double_vgg = double_vgg
        self.channel = channel
        self.double_vgg_share_param = double_vgg_share_param
        if self.double_vgg and not self.double_vgg_share_param:
            self.feature_extractor_T1, self.feature_extractor_DeepC = feature_extractor
        else:
            self.feature_extractor = feature_extractor

        #self.T1_feature_dimension = 1 * 128 * 5 * 6 * 5 # important, change this if the input is not 182x218x182
        self.T1_feature_dimension = 1 * 128 * 6 * 6 * 6 # important, change this if the input is not 64x77x64

        if (not double_vgg) and input_T1 and input_DeepC:
            # This case means: double channel for 2 inputs.
            self.DeepC_feature_dimension = self.T1_feature_dimension
        elif DeepC_isotropic:
            # Iso 1mm DeepC
            #self.DeepC_feature_dimension = 1 * 128 * 7 * 7 * 5
            self.DeepC_feature_dimension = 1 * 128 * 6 * 6 * 6

        else:
            # CUres DeepC
            self.DeepC_feature_dimension = 1 * 128 * 6 * 6 * 6

        if DeepC_isotropic_crop:
            # Iso 1mm DeepC after cropping
            self.DeepC_feature_dimension = 1 * 128 * 6 * 6 * 6

        if self.double_vgg:
            feature_dimension = self.T1_feature_dimension + self.DeepC_feature_dimension
            # self.se_block=SE_block(ch=256)
        elif self.input_DeepC:
            feature_dimension = self.DeepC_feature_dimension
        else:
            feature_dimension = self.T1_feature_dimension

        self.classifier = nn.Sequential(
            nn.Dropout(), # commented on 20201230
            nn.Linear(feature_dimension, 1024), #nn.Linear(feature_dimension, 2048),
            nn.ReLU(True),

            nn.Dropout(),
            nn.Linear(1024, 128), #nn.Linear(2048, 512),
            nn.Sigmoid(),
            #nn.ReLU(True),

            #nn.Dropout(),
            #nn.Linear(256, 64), #nn.Linear(2048, 512),
            #nn.ReLU(True),

            nn.Linear(128, 2), #nn.Linear(512, 2),
            #nn.Sigmoid()
         )

         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

        self.gradients = None
      
    def forward(self, x1, x2):
        if self.double_vgg:
            # This case means: double vgg encoder for 2 inputs.
            assert (x1 is not None) and (x2 is not None)
            if not self.double_vgg_share_param:
                x1 = self.feature_extractor_T1(x1)
                x2 = self.feature_extractor_DeepC(x2)
            else:
                x1 = self.feature_extractor(x1)
                x2 = self.feature_extractor(x2)
            # SE block to add content attention to T1 and ACBV
            if self.channel == 'T1':
                h = x1.register_hook(self.activations_hook)
            elif self.channel == 'DeepC':
                h = x2.register_hook(self.activations_hook)
            x = torch.cat((x1.view(x1.size(0), -1), x2.view(x2.size(0), -1)), dim = -1)
        elif self.input_T1 and self.input_DeepC:
            # This case means: double channel for 2 inputs.
            assert (x1 is not None) and (x2 is not None)
            x = torch.cat((x1, x2), dim = 1) # concatenate along channel.
            x = self.feature_extractor(x)
            h = x.register_hook(self.activations_hook)
            x = x.view(x.size(0), -1)
        elif self.input_T1:
            assert (x1 is not None) and (x2 is None)
            x1 = self.feature_extractor(x1)
            h = x1.register_hook(self.activations_hook)
            x = x1.view(x1.size(0), -1)
        elif self.input_DeepC:
            assert (x1 is None) and (x2 is not None)
            x2 = self.feature_extractor(x2)
            h = x2.register_hook(self.activations_hook)
            x = x2.view(x2.size(0), -1)

        x = self.classifier(x)
        return x

    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activation_gradient(self):
        return self.gradients
    
    def get_activations(self, x1, x2):
        if self.double_vgg:
            # This case means: double vgg encoder for 2 inputs.
            assert (x1 is not None) and (x2 is not None)
            if not self.double_vgg_share_param:
                x1 = self.feature_extractor_T1(x1)
                x2 = self.feature_extractor_DeepC(x2)
            else:
                x1 = self.feature_extractor(x1)
                x2 = self.feature_extractor(x2)

            if self.channel == 'DeepC':
                return x2
            elif self.channel == 'T1':
                return x1
        elif self.input_T1 and self.input_DeepC:
            # This case means: double channel for 2 inputs.
            assert (x1 is not None) and (x2 is not None)
            x = torch.cat((x1, x2), dim = 1) # concatenate along channel.
            x = self.feature_extractor(x)
            return x
        elif self.input_T1:
            assert (x1 is not None) and (x2 is None)
            x1 = self.feature_extractor(x1)
            return x1
        elif self.input_DeepC:
            assert (x1 is None) and (x2 is not None)
            x2 = self.feature_extractor(x2)
            return x2

# 5-27-2021
class SE_block(nn.Module):
    def __init__(self,ch,ratio=se_ratio):
        super(SE_block, self).__init__()

        # Both should use the same complex number for initiaization which is then split into real and imaginary parts
        # Weight initialiation using the real part 
        self.globalpooling = nn.AdaptiveAvgPool3d(1)
        # Weight initialization using the imag part
        self.linear1 = nn.Linear(ch,ch//ratio)
        self.relu1 = nn.ReLU()
        self.lienar2 = nn.Linear(ch//ratio,ch)
        self.sigmoid1 = nn.Sigmoid()
        self.ch = ch

    def forward(self, x):
        x_init=x
        #print('x shape: ',x.shape)
        x1=self.globalpooling(x).squeeze()
        x1=self.linear1(x1)
        x1=self.relu1(x1)
        x1=self.lienar2(x1)
        #print('self.ch shape: ',self.ch)
        x1=self.sigmoid1(x1).reshape(x.shape[0],self.ch,1,1,1)

        return torch.mul(x_init,x1)


def make_layers(cfg, input_T1, input_DeepC, double_vgg, double_vgg_share_param, batch_norm=False, se_block=False):
    layers = []
    DropoutRate = 0.10

    if input_T1 and input_DeepC and not double_vgg:
        in_channels = 2
    else:
        in_channels = 1

    if double_vgg_share_param or not double_vgg:
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool3d(kernel_size=2, stride=2)] # MaxPool without Dropout # commented on 20201230
                #layers += [nn.Dropout(p=DropoutRate), nn.MaxPool3d(kernel_size=2, stride=2)] # MaxPool with Dropout rate of 0.25 # commented on 20201230
            else:
                if batch_norm:
                    if se_block:
                        conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
                        se3d = SE_block(ch=v)
                        layers += [conv3d, nn.BatchNorm3d(v), se3d, nn.ReLU(inplace=True)]
                    else:
                        conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
                        layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv3d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    else:
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool3d(kernel_size=2, stride=2)] # MaxPool without Dropout # commented on 20201230
                #layers += [nn.Dropout(p=DropoutRate), nn.MaxPool3d(kernel_size=2, stride=2)] # MaxPool with Dropout rate of 0.25 # commented on 20201230
            else:
                if batch_norm:
                    if se_block:
                        conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
                        se3d = SE_block(ch=v)
                        layers += [conv3d, nn.BatchNorm3d(v), se3d, nn.ReLU(inplace=True)]
                    else:
                        conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
                        layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv3d, nn.ReLU(inplace=True)]
                in_channels = v
                #print(in_channels)
        feature_extractor_T1 = nn.Sequential(*layers)

        # Calculate the in_channel again for the second construction.
        layers = []
        if input_T1 and input_DeepC and not double_vgg:
            in_channels = 2
        else:
            in_channels = 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool3d(kernel_size=2, stride=2)] # MaxPool without Dropout # commented on 20201230
                #layers += [nn.Dropout(p=DropoutRate), nn.MaxPool3d(kernel_size=2, stride=2)] # MaxPool with Dropout rate of 0.25 # commented on 20201230
            else:
                if batch_norm:
                    if se_block:
                        conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
                        se3d = SE_block(ch=v)
                        layers += [conv3d, nn.BatchNorm3d(v), se3d, nn.ReLU(inplace=True)]
                    else:
                        conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=1)
                        layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv3d, nn.ReLU(inplace=True)]
                in_channels = v
        feature_extractor_DeepC = nn.Sequential(*layers)
        return feature_extractor_T1, feature_extractor_DeepC
          

cfg = {
    'A': [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128],
}

def vgg11_bn(input_T1, input_DeepC, double_vgg, double_vgg_share_param, DeepC_isotropic, DeepC_isotropic_crop, channel = None):
    """VGG 19-layer model (configuration 'A') with batch normalization"""
    return VGG(make_layers(cfg['A'], input_T1, input_DeepC, double_vgg, double_vgg_share_param, batch_norm=True, se_block=True), \
        input_T1, input_DeepC, double_vgg, double_vgg_share_param, DeepC_isotropic, DeepC_isotropic_crop, channel)

def center_crop_or_pad(input_scan, desired_dimension):
    input_dimension = input_scan.shape
    #print('Input dimension: ', input_dimension, '\ndesired dimension: ', desired_dimension)

    x_lowerbound_target = int(np.floor((desired_dimension[0] - input_dimension[0]) / 2)) if desired_dimension[0] >= input_dimension[0] else 0
    y_lowerbound_target = int(np.floor((desired_dimension[1] - input_dimension[1]) / 2)) if desired_dimension[1] >= input_dimension[1] else 0
    z_lowerbound_target = int(np.floor((desired_dimension[2] - input_dimension[2]) / 2)) if desired_dimension[2] >= input_dimension[2] else 0
    x_upperbound_target = x_lowerbound_target + input_dimension[0] if desired_dimension[0] >= input_dimension[0] else None
    y_upperbound_target = y_lowerbound_target + input_dimension[1] if desired_dimension[1] >= input_dimension[1] else None
    z_upperbound_target = z_lowerbound_target + input_dimension[2] if desired_dimension[2] >= input_dimension[2] else None

    x_lowerbound_input = 0 if desired_dimension[0] >= input_dimension[0] else int(np.floor((input_dimension[0] - desired_dimension[0]) / 2))
    y_lowerbound_input = 0 if desired_dimension[1] >= input_dimension[1] else int(np.floor((input_dimension[1] - desired_dimension[1]) / 2))
    z_lowerbound_input = 0 if desired_dimension[2] >= input_dimension[2] else int(np.floor((input_dimension[2] - desired_dimension[2]) / 2))
    x_upperbound_input = None if desired_dimension[0] >= input_dimension[0] else x_lowerbound_input + desired_dimension[0]
    y_upperbound_input = None if desired_dimension[1] >= input_dimension[1] else y_lowerbound_input + desired_dimension[1]
    z_upperbound_input = None if desired_dimension[2] >= input_dimension[2] else z_lowerbound_input + desired_dimension[2]


    output_scan = np.zeros(desired_dimension).astype(np.float32)

    output_scan[x_lowerbound_target : x_upperbound_target, \
                y_lowerbound_target : y_upperbound_target, \
                z_lowerbound_target : z_upperbound_target] = \
    input_scan[x_lowerbound_input: x_upperbound_input, \
               y_lowerbound_input: y_upperbound_input, \
               z_lowerbound_input: z_upperbound_input]

    return output_scan

if __name__ == '__main__':
    main()
