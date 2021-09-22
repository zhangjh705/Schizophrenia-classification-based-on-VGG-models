# Schizophrenia-classification-based-on-VGG-models

# Necessary packages in environment

### -- numpy  -- sklearn  -- torch  -- scipy
### -- tqdm  -- shutil  -- os


# The file VGG_model contains

### main_10fold.py: train and validate the model
### data_loader_10fold: load and extract the data for rhe model
### test.py: test the trained model on the test dataset
### generate_activation_map: generate the class activation map 

### run the command to train the model: 
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_MN152Affine_WB_3T_Batchx5_T1mean_lr1e-4_20210629_SE_block_ratio16_811_V-T_fold1-3_Adam_model_info --arch='vgg11_bn' --batch-size=5 --lr=1e-4 --cuda-idx=1 --data-dropout=False --input-T1=True --input-DeepC=False --T1-normalization-method=mean --val-folder='fold1' --test-folder='fold3' |& tee -a ./result/log/log_save_vgg11_bn_T1_MN152Affine_WB_3T_Batchx5_T1mean_lr1e-4_20210629_SE_block_ratio16_811_V-T_fold1-3_Adam_model_info 


### run the command to test the model in test dataset:
python test.py --arch='vgg11_bn' --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_MN152Affine_WB_3T_DSx2_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210425_RAW_fold10_MCIC --load-dir=./result/model/save_vgg11_bn_T1_MN152Affine_WB_3T_DSx2_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210425_RAW_fold10 --lr=0.0001 --cuda-idx=2 --data-dropout=False --input-T1=True --input-DeepC=False --T1-normalization-method=mean

### run the command to generate the class activation map:







# Citation


# Pretrained model and data

if you are going to use the pretrained model and the data for test, please contact jg3400@columbia.edu and get access to these sources.
