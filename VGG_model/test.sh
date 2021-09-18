cd /media/sail/HHD8T/DeepGd/Functional_Lesion/DeepC_Schizconnect/Schiz_classification_data_and_script/downsample_T1affine_VGG11_DCSE_311_new/

# T1
python test.py --arch='vgg11_bn' --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_MN152Affine_WB_3T_DSx2_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210425_RAW_fold10_MCIC --load-dir=./result/model/save_vgg11_bn_T1_MN152Affine_WB_3T_DSx2_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210425_RAW_fold10 --lr=0.0001 --cuda-idx=2 --data-dropout=False --input-T1=True --input-DeepC=False --T1-normalization-method=mean


python test_NMorph.py --arch='vgg11_bn' --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_MN152Affine_WB_3T_DSx2_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210425_RAW_fold10_MCIC --load-dir=./result/model/save_vgg11_bn_T1_MN152Affine_WB_3T_DSx2_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210425_RAW_fold10 --lr=0.0001 --cuda-idx=2 --data-dropout=False --input-T1=True --input-DeepC=False --T1-normalization-method=mean


python generate_activation_map.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_affine_DSx4_MB1000_T1max_feature16-32-64-128_lr0d0001_20201202 --load-dir=./result/model/save_vgg11_bn_T1_affine_DSx4_MB1000_T1max_feature16-32-64-128_lr0d0001_20201202 --lr=0.0001 --cuda-idx=2 --data-dropout=False  --input-T1=True --input-DeepC=False --T1-normalization-method=max --channel=T1

# DeepC
python test.py --arch='vgg11_bn' --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_DeepC_diffeoISO_DSx4_DeepCTop10_lr0d0001_useAUC_20201130 --load-dir=./result/model/save_vgg11_bn_DeepC_diffeoISO_DSx4_DeepCTop10_lr0d0001_useAUC_20201130 --lr=0.0001 --cuda-idx=2 --data-dropout=False --input-T1=False --input-DeepC=True --DeepC-isotropic=True --DeepC-normalization-method=WBtop10PercentMean

python generate_activation_map.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_DeepC_diffeoISO_DSx4_DeepCTop10_lr0d0001_useAUC_20201130 --load-dir=./result/model/save_vgg11_bn_DeepC_diffeoISO_DSx4_DeepCTop10_lr0d0001_useAUC_20201130 --lr=0.0001 --cuda-idx=2 --data-dropout=False  --input-T1=False --input-DeepC=True --DeepC-isotropic=True --DeepC-normalization-method=WBtop10PercentMean --channel=DeepC

# Double channel
python test.py --arch='vgg11_bn' --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_affine_DeepC_diffeoISOcrop_DSx4_doubleVGG_DiffWeights_T1maxDeepCTop10_lr0d0001_useAUC_20201201 --load-dir=./result/model/save_vgg11_bn_T1_affine_DeepC_diffeoISOcrop_DSx4_doubleVGG_DiffWeights_T1maxDeepCTop10_lr0d0001_useAUC_20201201 --lr=0.0001 --cuda-idx=2 --data-dropout=False --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --DeepC-normalization-method=WBtop10PercentMean --T1-normalization-method=max
python generate_activation_map.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_affine_DeepC_diffeoISOcrop_DSx4_doubleVGG_DiffWeights_T1maxDeepCTop10_lr0d0001_useAUC_20201201 --load-dir=./result/model/save_vgg11_bn_T1_affine_DeepC_diffeoISOcrop_DSx4_doubleVGG_DiffWeights_T1maxDeepCTop10_lr0d0001_useAUC_20201201 --lr=0.0001 --cuda-idx=2 --data-dropout=False  --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --DeepC-normalization-method=WBtop10PercentMean --T1-normalization-method=max --channel=DeepC
