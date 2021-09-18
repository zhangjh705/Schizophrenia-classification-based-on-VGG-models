### test code for different layer CAM (layer='1'/'2'/'3'/'4'/'5')

save_vgg11_bn_T1_MN152Affine_WB_3T_Batchx5_T1mean_lr1e-4_20210630_SE_block_ratio16_811_V-T_fold2-9_Adam



python generate_activation_map_test.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_MN152Affine_WB_3T_Batchx5_T1mean_lr1e-4_20210630_SE_block_ratio16_811_V-T_fold2-9_Adam --load-dir=./result/model/save_vgg11_bn_T1_MN152Affine_WB_3T_Batchx5_T1mean_lr1e-4_20210630_SE_block_ratio16_811_V-T_fold2-9_Adam --lr=1e-4 --cuda-idx=2 --data-dropout=False --input-T1=True --input-DeepC=False --T1-normalization-method=mean --DeepC-isotropic=True  --channel='T1' --DeepC-normalization-method=WBtop10PercentMean --double-vgg=False --test-folder='fold9' --layer='5'

python generate_activation_map_test.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_MN152Affine_WB_3T_Batchx5_T1mean_lr1e-4_20210630_SE_block_ratio16_811_V-T_fold2-9_Adam --load-dir=./result/model/save_vgg11_bn_T1_MN152Affine_WB_3T_Batchx5_T1mean_lr1e-4_20210630_SE_block_ratio16_811_V-T_fold2-9_Adam --lr=1e-4 --cuda-idx=2 --data-dropout=False --input-T1=True --input-DeepC=False --T1-normalization-method=mean --DeepC-isotropic=True  --channel='T1' --DeepC-normalization-method=WBtop10PercentMean --double-vgg=False --test-folder='fold9' --layer='3'

python generate_activation_map_test.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_MN152Affine_WB_3T_Batchx5_T1mean_lr1e-4_20210630_SE_block_ratio16_811_V-T_fold2-9_Adam --load-dir=./result/model/save_vgg11_bn_T1_MN152Affine_WB_3T_Batchx5_T1mean_lr1e-4_20210630_SE_block_ratio16_811_V-T_fold2-9_Adam --lr=1e-4 --cuda-idx=2 --data-dropout=False --input-T1=True --input-DeepC=False --T1-normalization-method=mean --DeepC-isotropic=True  --channel='T1' --DeepC-normalization-method=WBtop10PercentMean --double-vgg=False --test-folder='fold9' --layer='1'
