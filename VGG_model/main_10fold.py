import argparse
import os
import shutil
import time
import numpy as np
from sklearn.utils.multiclass import type_of_target
from itertools import chain

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

from sklearn.metrics import roc_curve, auc
import scipy

from tqdm import tqdm

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
                    metavar='N', help='batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
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
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='use cpu')
parser.add_argument('--cuda-idx', default='0', type=str,
                    help='cuda index')
parser.add_argument('--adaptive-lr', default=False, type=str2bool,
                    help='use adaptive learning rate or not')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
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
parser.add_argument('--save-all-models', default = False, type = str2bool,

                    help = 'Save all models or only save the best one?')
parser.add_argument('--double-vgg', default=True, type=str2bool,
                    help='Use two vgg encoder or use two channels. Only relevant when having two inputs.')
parser.add_argument('--double-vgg-share-param', default = True, type = str2bool,
                    help = 'Do we want the double VGG encoding layers to share parameters?')
parser.add_argument('--val-folder', default = 'fold2', type = str,
                    help = 'which folder is validation dataset?')
parser.add_argument('--test-folder', default = 'fold3', type = str,
                    help = 'which folder is test dataset?')

best_acc = 0
best_AUC = 0

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

    print('Batch size? : ', args.batch_size)
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
    print('saving directory: ', args.save_dir)
    print('validation folder: ', args.val_folder)
    print('test folder: ', args.test_folder)

    device = torch.device(f"cuda:{args.cuda_idx}" if (torch.cuda.is_available()) else "cpu")

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = vgg.__dict__[args.arch](input_T1 = args.input_T1, input_DeepC = args.input_DeepC, \
        double_vgg = args.double_vgg, double_vgg_share_param = args.double_vgg_share_param, \
        DeepC_isotropic = args.DeepC_isotropic, DeepC_isotropic_crop = args.DeepC_isotropic_crop)
    print(model)
    #print(model.classifier)
    #print(sum(p.numel() for p in model.classifier.parameters() if p.requires_grad))
    if len(args.cuda_idx) > 1:
    	model.features = torch.nn.DataParallel(model.features)
    if args.cpu:
        model.cpu()
    else:
        model.to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc1']
            best_AUC = checkpoint['best_AUC1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # define dataset dir
    TrainDataDir = '/media/sail/HDD10T/DeepC_SCZ-Score/10Fold_Dataset/'
    ValidationDataDir = '/media/sail/HDD10T/DeepC_SCZ-Score/10Fold_Dataset/'
    TestDataDir = '/media/sail/HDD10T/DeepC_SCZ-Score/10Fold_Dataset/'
    # data_fold_list=['fold1','fold11']
    data_fold_list=['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10']

    val_index=args.val_folder
    val_list=[val_index]
    print("=> validation folder: ")
    print(*val_list)
    data_fold_list.remove(val_index)
    
    test_index=args.test_folder
    test_list=[test_index]
    print("=> test folder: ")
    print(*test_list)
    data_fold_list.remove(test_index)

    train_list=data_fold_list
    #train_list=val_list
    print("=> training folders: ")
    print(*train_list)

    # Train
    Train_MRIDataset = MRIDataset(DataDir = TrainDataDir, mode = 'train', input_T1 = args.input_T1, input_DeepC = args.input_DeepC, double_vgg = args.double_vgg,
                            DeepC_isotropic = args.DeepC_isotropic, DeepC_isotropic_crop = args.DeepC_isotropic_crop, transform=transforms.Compose([ToTensor()]),
                            T1_normalization_method = args.T1_normalization_method, DeepC_normalization_method = args.DeepC_normalization_method,fold_list=train_list)
    Train_dataloader = DataLoader(Train_MRIDataset, batch_size = args.batch_size,
                           shuffle=True, num_workers=args.workers)

    # Validation
    Validation_MRIDataset = MRIDataset(DataDir=ValidationDataDir, mode = 'validation', input_T1 = args.input_T1, input_DeepC = args.input_DeepC, double_vgg = args.double_vgg,
                                    DeepC_isotropic = args.DeepC_isotropic, DeepC_isotropic_crop = args.DeepC_isotropic_crop, transform=transforms.Compose([ToTensor()]),
                                    T1_normalization_method = args.T1_normalization_method, DeepC_normalization_method = args.DeepC_normalization_method,fold_list=val_list)
    Val_dataloader = DataLoader(Validation_MRIDataset, batch_size = args.batch_size,
                           shuffle=False, num_workers=args.workers)


    # Test
    Test_MRIDataset = MRIDataset(DataDir=TestDataDir, mode = 'test', input_T1 = args.input_T1, input_DeepC = args.input_DeepC, double_vgg = args.double_vgg,
                                    DeepC_isotropic = args.DeepC_isotropic, DeepC_isotropic_crop = args.DeepC_isotropic_crop, transform=transforms.Compose([ToTensor()]),
                                    T1_normalization_method = args.T1_normalization_method, DeepC_normalization_method = args.DeepC_normalization_method,fold_list=test_list)
    Test_dataloader = DataLoader(Test_MRIDataset, batch_size = args.batch_size,
                           shuffle=False, num_workers=args.workers)


    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.cpu:
        criterion = criterion.cpu()
    else:
        criterion = criterion.to(device)
     

    ## determine the params to optimize
    params_to_update=[]
    for name,params in model.named_parameters():
    	if params.requires_grad == True:
    		params_to_update.append(params)
    print('the params for optimization is determined !')

    # Initialize the checker for early stop
    early_stop_checker = EarlyStopChecker()

    # #optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    
    optimizer_Adam = torch.optim.Adam(params_to_update, args.lr)

    optimizer_SGD = torch.optim.SGD(params_to_update, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    if args.adaptive_lr == True:
        print('Initializing adaptive learning rate... ')
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.8, patience = 2, verbose = 2)
    else:
        scheduler = None

    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    train_data_list=[]
    val_data_list=[]
    test_data_list=[]
    for epoch in range(args.start_epoch, args.epochs):
        #adjust_learning_rate(optimizer, epoch)
        print('************************')
        print('current epoch=',epoch)

        # switch to SGD after epoch > ...
        if epoch <= 100:
            optimizer = optimizer_Adam 
            print('optimizer = optimizer_Adam')
        else:
            optimizer = optimizer_SGD
            print('optimizer = optimizer_SGD')

        #optimizer = optimizer_Adam
        #print('optimizer = optimizer_Adam')

        # gradient reduction after epoch > ...
        #if epoch == 20:
        #    args.lr = args.lr * 0.1

        #if epoch == 40:
        #    args.lr = args.lr * 0.1

        print('current lr=',args.lr)

        print('************************')

        acc,auc,sen,spe,loss=train(Train_dataloader, model, criterion, optimizer, epoch, device)
        temp=[loss,acc,sen,spe,auc]
        train_data_list.append(temp)
        # evaluate on validation set
        current_acc, current_AUC,sen,spe,loss = validate(Val_dataloader, model, criterion, device, scheduler)
        temp=[loss,current_acc,sen,spe,current_AUC]
        val_data_list.append(temp)

        # remember best acc@1 and save checkpoint
        is_best = best_acc < current_acc
        #is_best = best_AUC < current_AUC
        best_acc = max(current_acc, best_acc)
        best_AUC = max(current_AUC, best_AUC)

        if args.save_all_models or is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc,
                'best_AUC1': best_AUC,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))
        np.save(os.path.join(args.save_dir,'train_performance.npy'),train_data_list)
        np.save(os.path.join(args.save_dir,'val_performance.npy'),val_data_list)

        # evaluate on test set
        test_current_acc, test_current_AUC, test_sen, test_spe, test_loss = test(Test_dataloader, model, criterion, device, scheduler)
        temp=[test_loss,test_current_acc,test_sen,test_spe,test_current_AUC]
        test_data_list.append(temp)

        # early stopping disabled
        #if early_stop_checker(current_AUC):
        #    print('The performance on validation set has not improved for a while. Early stop. Training completed.')
        #    break

        if early_stop_checker(current_acc):
            print('The performance on validation set has not improved for a while. Early stop. Training completed.')
            print('=============================================================')
            break

        

def train(train_loader, model, criterion, optimizer, epoch, device):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    counter = 0
    train_display_counter = 0
    AD_prediction_array, AD_ground_truth_array = [], []
		
    # switch to train mode
    model.train()

    end = time.time()

    # If we use data-dropout, we randomly sample a certain subset for training in each epoch.
    if args.data_dropout:
        print('The size of the training set is :', len(train_loader), '. We will randomly sample %s scans for this epoch.' % (args.data_dropout_remaining_size))
        valid_training_indices = sorted(np.random.choice(range(len(train_loader)), args.data_dropout_remaining_size, replace = False))

    for i, (data) in enumerate(train_loader):
        counter = counter+1
        if args.data_dropout:
            if not i in valid_training_indices:
                continue

        # assign variables
        input_data_T1, input_data_DeepC = None, None
        if args.input_T1:
            input_data_T1 = data['T1'].unsqueeze(1)
        if args.input_DeepC:
            input_data_DeepC = data['DeepC'].unsqueeze(1)
        target = data['label']

        # measure data loading time
        data_time.update(time.time() - end)


        if args.cpu == False:
            if args.input_T1:
                input_data_T1 = input_data_T1.to(device)
            if args.input_DeepC:
                input_data_DeepC = input_data_DeepC.to(device)
            target = target.to(device)

        # compute output
        output = model(input_data_T1, input_data_DeepC)

        #print('output: ', output, 'target: ', target)
        
        l1_regularization, l2_regularization = 0, 0

        for param in model.parameters():
            l1_regularization += torch.norm(param,1)**2
            l2_regularization += torch.norm(param,2)**2
        
        #regu_lambda = 0.1 
        #loss = criterion(output, target) + regu_lambda**l1_regularization
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        #clip_grad_value_(model.parameters(), 0.25) # Gradient clipping.

        ## The following code for printing out gradient.
        #grads = []
        #for param in model.parameters():
        #   grads.append(param.grad.view(-1))
        #grads = torch.cat(grads)
        #grads = np.sqrt(np.sum((grads.cpu().detach().numpy())**2))
        #print('Model Gradient', grads)

        optimizer.step()

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        acc1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), args.batch_size)
        top1.update(acc1.item(), args.batch_size)

        # measure sensitivity, specificity, AUC.
        softmax1=nn.Softmax(dim=1)
        AD_prediction=softmax1(output.data)[:,-1]

        if counter==1:
            AD_prediction_array=AD_prediction.cpu().detach().numpy()
            AD_ground_truth_array=target.cpu().detach().numpy()
        else:
            AD_prediction_array=np.concatenate((AD_prediction_array,AD_prediction.cpu().detach().numpy()),axis=None)
            AD_ground_truth_array=np.concatenate((AD_ground_truth_array,target.cpu().detach().numpy()),axis=None)


        #print(type_of_target(AD_ground_truth_array))

        #AD_prediction_list = [scipy.special.softmax(pred)[0][1] for pred in model_output_list]

        #print('AD_ground_truth_array length',len(AD_ground_truth_array))
        #print('AD_prediction_array length',len(AD_prediction_array))
        #print('AD_ground_truth_array first 4',AD_ground_truth_array[0:4])
        #print('AD_prediction_array first 4',AD_prediction_array[0:4])

        fpr, tpr, _ = roc_curve(AD_ground_truth_array, AD_prediction_array)
        operating_point_index = np.argmax(1 - fpr + tpr)
        sensitivity, specificity = tpr[operating_point_index], 1 - fpr[operating_point_index]
        AUC = auc(fpr, tpr)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        torch.cuda.empty_cache()

        if args.data_dropout:
            if (i > (len(train_loader) // 3) and train_display_counter == 0) or \
            (i > (len(train_loader) * 2 // 3) and train_display_counter == 1):
                print('\nEpoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Sensitivity ({sensitivity:.3f})\t'
                      'Specificity ({specificity:.3f})\t'
                      'AUC ({AUC:.3f})'.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, top1=top1,
                          sensitivity=sensitivity, specificity=specificity, AUC=AUC))
                train_display_counter += 1
        else:
            if i % (len(train_loader) // 2) == 0:
                print('\nEpoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy@1 {top1.val:.3f} ({top1.avg:.3f})\t'

                      'Sensitivity ({sensitivity:.3f})\t'
                      'Specificity ({specificity:.3f})\t'
                      'AUC ({AUC:.3f})'.format(
                          epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, top1=top1,
                          sensitivity=sensitivity, specificity=specificity, AUC=AUC))
			   
    return top1.avg, AUC,sensitivity,specificity,losses.avg


def validate(val_loader, model, criterion, device, scheduler = None):
    """
    Run evaluation
    """
    counter = 0
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    AD_prediction_array, AD_ground_truth_array = [], []

    # switch to evaluate mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (data) in enumerate(val_loader):
            counter = counter+1
            # assign variables
            input_data_T1, input_data_DeepC = None, None
            if args.input_T1:
                input_data_T1 = data['T1'].unsqueeze(1)
            if args.input_DeepC:
                input_data_DeepC = data['DeepC'].unsqueeze(1)
            target = data['label']

            # measure data loading time
            batch_time.update(time.time() - end)

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
            softmax1=nn.Softmax(dim=1)
            AD_prediction=softmax1(output.data)[:,-1]

            if counter==1:
            	AD_prediction_array=AD_prediction.cpu().detach().numpy()
            	AD_ground_truth_array=target.cpu().detach().numpy()
            else:
            	AD_prediction_array=np.concatenate((AD_prediction_array,AD_prediction.cpu().detach().numpy()),axis=None)
            	AD_ground_truth_array=np.concatenate((AD_ground_truth_array,target.cpu().detach().numpy()),axis=None)

            fpr, tpr, _ = roc_curve(AD_ground_truth_array, AD_prediction_array)
            operating_point_index = np.argmax(1 - fpr + tpr)
            sensitivity, specificity = tpr[operating_point_index], 1 - fpr[operating_point_index]
            AUC = auc(fpr, tpr)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            torch.cuda.empty_cache()

            if i == len(val_loader):
                print('\nValidation: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Sensitivity ({sensitivity:.3f})\t'
                      'Specificity ({specificity:.3f})\t'
                      'AUC@1 ({AUC:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1, sensitivity=sensitivity, specificity=specificity, AUC=AUC))

        print('\n *** Validation Performance ***')
        print('\n * Accuracy@1 {top1.avg:.3f}'.format(top1=top1))
        print(' * Sensitivity {sensitivity:.3f} Specificity {specificity:.3f} AUC {AUC:.3f}'
            .format(sensitivity=sensitivity, specificity=specificity, AUC=AUC))

        if args.adaptive_lr == True:
                scheduler.step(loss)

        return top1.avg, AUC,sensitivity,specificity,losses.avg

def test(test_loader, model, criterion, device, scheduler = None):
    """
    Run test
    """

    counter = 0
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    AD_prediction_array, AD_ground_truth_array = [], []

    # switch to evaluate mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (data) in enumerate(test_loader):
            counter = counter+1
            # assign variables
            input_data_T1, input_data_DeepC = None, None
            if args.input_T1:
                input_data_T1 = data['T1'].unsqueeze(1)
            if args.input_DeepC:
                input_data_DeepC = data['DeepC'].unsqueeze(1)
            target = data['label']

            # measure data loading time
            batch_time.update(time.time() - end)

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
            softmax1=nn.Softmax(dim=1)
            AD_prediction=softmax1(output.data)[:,-1]

            if counter==1:
            	AD_prediction_array=AD_prediction.cpu().detach().numpy()
            	AD_ground_truth_array=target.cpu().detach().numpy()
            else:
            	AD_prediction_array=np.concatenate((AD_prediction_array,AD_prediction.cpu().detach().numpy()),axis=None)
            	AD_ground_truth_array=np.concatenate((AD_ground_truth_array,target.cpu().detach().numpy()),axis=None)

            fpr, tpr, _ = roc_curve(AD_ground_truth_array, AD_prediction_array)
            operating_point_index = np.argmax(1 - fpr + tpr)
            sensitivity, specificity = tpr[operating_point_index], 1 - fpr[operating_point_index]
            AUC = auc(fpr, tpr)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            torch.cuda.empty_cache()

            if i == len(test_loader):
                print('\nTesting: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Sensitivity ({sensitivity:.3f})\t'
                      'Specificity ({specificity:.3f})\t'
                      'AUC@1 ({AUC:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1, sensitivity=sensitivity, specificity=specificity, AUC=AUC))

        print('\n *** Testing Performance ***')
        print('\n * Accuracy@1 {top1.avg:.3f}'.format(top1=top1))
        print(' * Sensitivity {sensitivity:.3f} Specificity {specificity:.3f} AUC {AUC:.3f}'
            .format(sensitivity=sensitivity, specificity=specificity, AUC=AUC))

        return top1.avg, AUC,sensitivity,specificity,losses.avg

class EarlyStopChecker:
    '''
    Early stop checker. Credits to Chen "Raphael" Liu and Nanyan "Rosalie" Zhu.
    '''
    def __init__(self, search_window = 10, score_higher_better = True):
        self.search_window = search_window
        self.score_higher_better = score_higher_better
        self.score_history = []

    def __call__(self, current_score):
        self.score_history.append(current_score)
        if len(self.score_history) < 5 * self.search_window:
            return False
        else:
            if self.score_higher_better == True:
                if current_score < np.max(self.score_history) and np.mean(self.score_history[-self.search_window:]) < np.mean(self.score_history[-2*self.search_window:-self.search_window]):

                    return True
                else:
                    return False
            else:
                if current_score > np.max(self.score_history) and np.mean(self.score_history[-self.search_window:]) > np.mean(self.score_history[-2*self.search_window:-self.search_window]):
                    return True
                else:
                    return False

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """

    Save the training model

    """
    
    # save the best model
    if is_best:
        best_filename = '/'.join(filename.split('/')[:-1]) + '/checkpoint_best.tar'
        torch.save(state, best_filename)
        print('=============================================================')
        print('Using ACC as the criterion for selecting best model. A new best model appears here.')
        print('=============================================================')
    else:
    	torch.save(state, filename)

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


#def adjust_learning_rate(optimizer, epoch):
#    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
#    lr = args.lr * (0.5 ** (epoch // 30))
#    for param_group in optimizer.param_groups:
#        param_group['lr'] = lr


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
