# Schizophrenia-classification-based-on-VGG-models

## Necessary packages in environment

-- numpy  -- sklearn  -- torch  -- scipy
-- tqdm  -- shutil  -- os


## VGG_model folder 

#### main_10fold.py: train and validate the model
#### data_loader_10fold: load and extract the data for rhe model
#### test.py: test the trained model on the test dataset 
#### generate_activation_map: generate the class activation map 

### run the command to train the model: 

python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_MN152Affine_WB_3T_Batchx5_T1mean_lr1e-4_20210629_SE_block_ratio16_811_V-T_fold1-3_Adam_model_info --arch='vgg11_bn' --batch-size=5 --lr=1e-4 --cuda-idx=1 --data-dropout=False --input-T1=True --input-DeepC=False --T1-normalization-method=mean --val-folder='fold1' --test-folder='fold3' |& tee -a ./result/log/log_save_vgg11_bn_T1_MN152Affine_WB_3T_Batchx5_T1mean_lr1e-4_20210629_SE_block_ratio16_811_V-T_fold1-3_Adam_model_info 

### parameter list illustration:
### save-dir: the directory for model saving
### arch: the model architecture
### input-T1: whether T1 structure MRI brain scans are used
### input-DeepC: whether artificial CBV brain maps are used
### val-folder: the data folder used for validation in training
### test-folder: the data foldr used for test after model training finished

### run the command to test the model in test dataset:

python test.py --arch='vgg11_bn' --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_MN152Affine_WB_3T_DSx2_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210425_RAW_fold10_MCIC --load-dir=./result/model/save_vgg11_bn_T1_MN152Affine_WB_3T_DSx2_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210425_RAW_fold10 --lr=0.0001 --cuda-idx=2 --data-dropout=False --input-T1=True --input-DeepC=False --T1-normalization-method=mean

### run the command to generate the class activation map:

python generate_activation_map_test.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_MN152Affine_WB_3T_Batchx5_T1mean_lr1e-4_20210630_SE_block_ratio16_811_V-T_fold2-9_Adam --load-dir=./result/model/save_vgg11_bn_T1_MN152Affine_WB_3T_Batchx5_T1mean_lr1e-4_20210630_SE_block_ratio16_811_V-T_fold2-9_Adam --lr=1e-4 --cuda-idx=2 --data-dropout=False --input-T1=True --input-DeepC=False --T1-normalization-method=mean --DeepC-isotropic=True  --channel='T1' --DeepC-normalization-method=WBtop10PercentMean --double-vgg=False --test-folder='fold9' --layer='5

## Citation


## Pretrained model and data

if you are going to use the pretrained model and the data for test, please contact jg3400@columbia.edu and get access to these sources.
