cd /media/sail/HHD8T/DeepGd/Functional_Lesion/DeepC_Schizconnect/Schiz_classification_data_and_script/downsample_T1affine_VGG11_DCSE_311_new/

# upload T1_affine_DeepC_ISO_generate_activation_map.py

### T1 affine activation map

## check the data_loader T1_input: T1_affine_input path

python T1_affine_DeepC_ISO_generate_activation_map.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_affine_DSx4_MB1000_T1max_feature16-32-64-128_lr0d0001_epoch100_20201124 --load-dir=./result/model/save_vgg11_bn_T1_affine_DSx4_MB1000_T1max_feature16-32-64-128_lr0d0001_epoch100_20201124 --lr=0.0001 --cuda-idx=2 --data-dropout=False --input-T1=True --input-DeepC=False --T1-normalization-method=max --channel=T1


## Deepc ISO activation map

# Note: check the Deep_input path : DeepC_iso folder in data_loader.py

python T1_affine_DeepC_ISO_generate_activation_map.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_DeepC_diffeoISO_DSx4_DeepCTop10_lr0d0001_useAUC_20201203_9pm --load-dir=./result/model/save_vgg11_bn_DeepC_diffeoISO_DSx4_DeepCTop10_lr0d0001_useAUC_20201203_9pm --lr=0.0001 --cuda-idx=2 --data-dropout=False --input-T1=False --input-DeepC=True --channel=DeepC --DeepC-normalization-method=WBtop10PercentMean --DeepC-isotropic=True


## T1_diffeo_CU_ISO actrivation map

# Note : change the Deepc_input path from Deepc_iso folder to T1_diffeo_ISO folder in data_loader.py

python T1_affine_DeepC_ISO_generate_activation_map.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_DeepC_diffeoISO_DSx4_DeepCTop10_lr0d0001_useAUC_20201202 --load-dir=./result/model/save_vgg11_bn_DeepC_diffeoISO_DSx4_DeepCTop10_lr0d0001_useAUC_20201202 --lr=0.0001 --cuda-idx=2 --data-dropout=False --input-T1=False --input-DeepC=True --channel=DeepC --DeepC-normalization-method=WBtop10PercentMean --DeepC-isotropic=True
