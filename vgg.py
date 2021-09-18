'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch
import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg6_bn', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

# set random seed
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# define global parameters
se_ratio = 16

class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, feature_extractor, input_T1, input_DeepC, double_vgg, double_vgg_share_param, DeepC_isotropic, DeepC_isotropic_crop):
        super(VGG, self).__init__()
        self.input_T1 = input_T1
        self.input_DeepC = input_DeepC

        self.double_vgg = double_vgg
        self.double_vgg_share_param = double_vgg_share_param
        if self.double_vgg and not self.double_vgg_share_param:
            self.feature_extractor_T1, self.feature_extractor_DeepC = feature_extractor
        else:
            self.feature_extractor = feature_extractor

        #self.T1_feature_dimension = 1 * 128 * 5 * 6 * 5 # important, change this if the input is not 182x218x182
        # self.T1_feature_dimension = 1 * 128 * 3 * 3 * 3 #  downsample 100*100*100 as T1 input # important, change this if the input is not 64x77x64
        self.T1_feature_dimension = 1 * 128 * 6 * 6 * 6 # raw T1 data 200*200*200
        
        if (not double_vgg) and input_T1 and input_DeepC:
            # This case means: double channel for 2 inputs.
            self.DeepC_feature_dimension = self.T1_feature_dimension
        elif DeepC_isotropic:
            # Iso 1mm DeepC
            self.DeepC_feature_dimension = 1 * 128 * 6 * 6 * 6 #1 * 64 * 8 * 9 * 8 # important, change this if the input is not ?
        else:
            # CUres DeepC
            self.DeepC_feature_dimension = 1 * 128 * 3 * 3 * 3 # important, change this if the input is not ?
        if DeepC_isotropic_crop:
            # Iso 1mm DeepC after cropping
            self.DeepC_feature_dimension = 1 * 128 * 3 * 3 * 3 # important, change this if the input is not ?

        if self.double_vgg:
            feature_dimension = self.T1_feature_dimension + self.DeepC_feature_dimension
            self.se_block = SE_block(ch=256)
            
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

        #self.classifier = nn.Sequential(
        #    nn.Dropout(), # commented on 20201230
        #    nn.Linear(feature_dimension, 512), #nn.Linear(feature_dimension, 2048),
        #    nn.ReLU(True),
        #    nn.Dropout(),
        #    nn.Linear(512, 256), #nn.Linear(2048, 512),
        #    nn.ReLU(True),
        #    nn.Linear(256, 2), #nn.Linear(512, 2),
        # )

         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


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
            x = torch.cat((x1,x2),dim = 1)
            #print('x shape: ',x.shape)
            x = self.se_block(x)
            #print('x shape: ',x.shape)
            x = x.view(x.size(0), -1)
        elif self.input_T1 and self.input_DeepC:
            # This case means: double channel for 2 inputs.
            assert (x1 is not None) and (x2 is not None)
            x = torch.cat((x1, x2), dim = 1) # concatenate along channel.
            x = self.feature_extractor(x)
            x = x.view(x.size(0), -1)
        elif self.input_T1:
            assert (x1 is not None) and (x2 is None)
            x1 = self.feature_extractor(x1)
            x = x1.view(x1.size(0), -1)
        elif self.input_DeepC:
            assert (x1 is None) and (x2 is not None)
            x2 = self.feature_extractor(x2)
            x = x2.view(x2.size(0), -1)

        x = self.classifier(x)
        return x

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
    'A': [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128], #'A': [32, 'M', 64, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M'], 
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 64, 'M', 128, 128, 128, 128, 'M', 
          128, 128, 128, 128, 'M'],
    'F': [32, 'M', 128, 'M', 512, 'M'],
}

def vgg19_bn(input_T1, input_DeepC, double_vgg, double_vgg_share_param, DeepC_isotropic, DeepC_isotropic_crop):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], input_T1, input_DeepC, double_vgg, double_vgg_share_param, batch_norm=True, se_block=True), \
        input_T1, input_DeepC, double_vgg, double_vgg_share_param, DeepC_isotropic, DeepC_isotropic_crop)

def vgg16_bn(input_T1, input_DeepC, double_vgg, double_vgg_share_param, DeepC_isotropic, DeepC_isotropic_crop):
    """VGG 16-layer model (configuration 'D') with batch normalization"""
    return VGG(make_layers(cfg['D'], input_T1, input_DeepC, double_vgg, double_vgg_share_param, batch_norm=True, se_block=True), \
        input_T1, input_DeepC, double_vgg, double_vgg_share_param, DeepC_isotropic, DeepC_isotropic_crop)

def vgg11_bn(input_T1, input_DeepC, double_vgg, double_vgg_share_param, DeepC_isotropic, DeepC_isotropic_crop):
    """VGG 11-layer model (configuration 'A') with batch normalization"""
    return VGG(make_layers(cfg['A'], input_T1, input_DeepC, double_vgg, double_vgg_share_param, batch_norm=True, se_block=True), \
        input_T1, input_DeepC, double_vgg, double_vgg_share_param, DeepC_isotropic, DeepC_isotropic_crop)

def vgg6_bn(input_T1, input_DeepC, double_vgg, double_vgg_share_param, DeepC_isotropic, DeepC_isotropic_crop):
    """VGG 8-layer model (configuration 'F') with batch normalization"""
    return VGG(make_layers(cfg['F'], input_T1, input_DeepC, double_vgg, double_vgg_share_param, batch_norm=True, se_block=True), \
        input_T1, input_DeepC, double_vgg, double_vgg_share_param, DeepC_isotropic, DeepC_isotropic_crop)

'''
def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))
'''
