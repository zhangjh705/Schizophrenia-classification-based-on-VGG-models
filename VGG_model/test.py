import argparse
import os
import shutil
import time
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
import vgg
from tqdm import tqdm

from collections import OrderedDict

from sklearn.metrics import roc_curve, auc, accuracy_score
import scipy

from data_loader_10fold import MRIDataset

model_names = sorted(name for name in vgg.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("vgg")
                     and callable(vgg.__dict__[name]))

# set random seed
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19_bn',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: vgg19)')
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
parser.add_argument('--testlist', help='Which folder to load for testing', default='fold3', type=str)

class ToTensor(object):
    def __call__(self, sample):
        torch_sample = {}
        for key, value in sample.items():
        	if key == 'label':
        		torch_sample[key] = torch.from_numpy(np.array(value))
        	else:
        		torch_sample[key] = torch.from_numpy(value)

        return torch_sample

def main():
    global args, best_acc, best_AUC
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
    print('We will be running test from this directory: ', args.testlist)

    device = torch.device(f"cuda:{args.cuda_idx}" if (torch.cuda.is_available()) else "cpu")

    model = vgg.__dict__[args.arch](input_T1 = args.input_T1, input_DeepC = args.input_DeepC, \
        double_vgg = args.double_vgg, double_vgg_share_param = args.double_vgg_share_param, \
        DeepC_isotropic = args.DeepC_isotropic, DeepC_isotropic_crop = args.DeepC_isotropic_crop)

    #print(model.classifier)
    #print(sum(p.numel() for p in model.classifier.parameters() if p.requires_grad))
    if len(args.cuda_idx) > 1:
    	model.features = torch.nn.DataParallel(model.features)
    if args.cpu:
        model.cpu()
    else:
        model.to(device)

    # define dataset dir
    test_list=[args.testlist]

    TestDataDir = '/media/sail/HDD10T/DeepC_SCZ-Score/10Fold_Dataset/'
    Test_MRIDataset = MRIDataset(DataDir = TestDataDir, mode = 'test', input_T1 = args.input_T1, input_DeepC = args.input_DeepC, double_vgg = args.double_vgg,
                            DeepC_isotropic = args.DeepC_isotropic, DeepC_isotropic_crop = args.DeepC_isotropic_crop, transform=transforms.Compose([ToTensor()]),
                            T1_normalization_method = args.T1_normalization_method, DeepC_normalization_method = args.DeepC_normalization_method,fold_list=test_list)
    Test_dataloader = DataLoader(Test_MRIDataset, batch_size = args.batch_size,
                           shuffle=False, num_workers=4)

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.cpu:
        criterion = criterion.cpu()
    else:
        criterion = criterion.to(device)

    # Load the saved model.
    checkpoint_file = args.load_dir + '/checkpoint_%s.tar' % args.which_to_load

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

    test(Test_dataloader, model, criterion, device)


def test(test_loader, model, criterion, device):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    model_output_list, AD_ground_truth_list = [], []

    # switch to evaluate mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (data) in enumerate(tqdm(test_loader)):
            
            # assign variables
            input_data_T1, input_data_DeepC = None, None
            if args.input_T1:
                input_data_T1 = data['T1'].unsqueeze(1)
            if args.input_DeepC:
                input_data_DeepC = data['DeepC'].unsqueeze(1)
            target = data['label']

            if args.cpu == False:
                if args.input_T1:
                    input_data_T1 = input_data_T1.to(device)
                if args.input_DeepC:
                    input_data_DeepC = input_data_DeepC.to(device)
                target = target.to(device)

            # compute output
            with torch.no_grad():
                output = model(input_data_T1, input_data_DeepC)
                loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            acc1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), args.batch_size)
            top1.update(acc1.item(), args.batch_size)

            # measure sensitivity, specificity, AUC.
            model_output_list.append(output.data.cpu().detach().numpy().tolist())
            AD_ground_truth_list.append(target.cpu().detach().numpy().tolist())

            AD_prediction_list = [scipy.special.softmax(pred)[0][1] for pred in model_output_list]
            AD_prob_gt_CN_list = [int(scipy.special.softmax(pred)[0][1] > scipy.special.softmax(pred)[0][0]) for pred in model_output_list]
            fpr, tpr, thresholds = roc_curve(AD_ground_truth_list, AD_prediction_list)
            operating_point_index = np.argmax(1 - fpr + tpr)
            sensitivity, specificity = tpr[operating_point_index], 1 - fpr[operating_point_index]
            AUC = auc(fpr, tpr)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            torch.cuda.empty_cache()

            if i == len(test_loader):
                print('Validation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Sensitivity ({sensitivity:.3f})\t'
                      'Specificity ({specificity:.3f})\t'
                      'AUC@1 ({AUC:.3f})'.format(
                          i, len(test_loader), batch_time=batch_time, loss=losses,
                          top1=top1, sensitivity=sensitivity, specificity=specificity, AUC=AUC))

        ACC_0d5 = accuracy_score(AD_ground_truth_list, list(np.array(AD_prediction_list) > 0.5))
        ACC_operating_point = accuracy_score(AD_ground_truth_list, AD_prediction_list > thresholds[operating_point_index])
        ACC_highest = np.max([accuracy_score(AD_ground_truth_list, AD_prediction_list > thresholds[index]) for index in range(len(thresholds))])
        ACC_model_raw = accuracy_score(AD_ground_truth_list, AD_prob_gt_CN_list)
        
        print(' * Accuracy@1 {top1.avg:.3f}'.format(top1=top1))
        print(' * Sensitivity {sensitivity:.3f} Specificity {specificity:.3f} AUC {AUC:.3f} \
              \nACC raw {ACC_model_raw:.3f} ACC @thr=0.5 {ACC_0d5:.3f} ACC @thr=operating {ACC_operating_point:.3f} ACC max {ACC_highest:.3f}'
            .format(sensitivity=sensitivity, specificity=specificity, AUC=AUC, \
                ACC_model_raw = ACC_model_raw, ACC_0d5 = ACC_0d5, ACC_operating_point = ACC_operating_point, ACC_highest = ACC_highest))

        os.makedirs(args.save_prediction_numpy_dir, exist_ok = True)
        np.save(args.save_prediction_numpy_dir + '/best_target.npy', AD_ground_truth_list)
        np.save(args.save_prediction_numpy_dir + '/best_prediction.npy', AD_prediction_list)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
