## T1 and ACBV CAM

python generate_activation_map_test.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_5e-5_T1mean_20210630_SE_block_ratio16_811_V-T_fold2-9_Adam --load-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_5e-5_T1mean_20210630_SE_block_ratio16_811_V-T_fold2-9_Adam --lr=1e-4 --cuda-idx=0 --data-dropout=False --input-T1=True --input-DeepC=True --T1-normalization-method=mean --DeepC-isotropic=True  --channel='T1' --DeepC-normalization-method=WBtop10PercentMean --double-vgg=True --layer='5' --test-folder='fold9'

python generate_activation_map_test.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_5e-5_T1mean_20210630_SE_block_ratio16_811_V-T_fold2-9_Adam --load-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_5e-5_T1mean_20210630_SE_block_ratio16_811_V-T_fold2-9_Adam --lr=1e-4 --cuda-idx=0 --data-dropout=False --input-T1=True --input-DeepC=True --T1-normalization-method=mean --DeepC-isotropic=True  --channel='DeepC' --DeepC-normalization-method=WBtop10PercentMean --double-vgg=True --layer='5' --test-folder='fold9'

# 811, T1 and ACBV, SE_block, SE_block, test 07/02/21 FINAL
python test.py --arch='vgg11_bn' --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_5e-5_T1mean_20210630_SE_block_ratio16_811_V-T_fold2-9_Adam --load-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_5e-5_T1mean_20210630_SE_block_ratio16_811_V-T_fold2-9_Adam --lr=0.0001 --cuda-idx=2 --data-dropout=False --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --DeepC-normalization-method=WBtop10PercentMean --T1-normalization-method=mean --testlist='fold9'
###################
 * Accuracy@1 96.000
 * Sensitivity 1.000 Specificity 0.917 AUC 0.990               
ACC raw 0.960 ACC @thr=0.5 0.960 ACC @thr=operating 0.947 ACC max 0.947


python test.py --arch='vgg11_bn' --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_T1mean_20210630_SE_block_ratio16_811_V-T_fold3-9_Adam --load-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_T1mean_20210630_SE_block_ratio16_811_V-T_fold3-9_Adam --lr=0.0001 --cuda-idx=2 --data-dropout=False --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --DeepC-normalization-method=WBtop10PercentMean --T1-normalization-method=mean --testlist='fold9'
####################
 * Accuracy@1 93.333
 * Sensitivity 1.000 Specificity 0.889 AUC 0.979               
ACC raw 0.933 ACC @thr=0.5 0.933 ACC @thr=operating 0.933 ACC max 0.933



python test.py --arch='vgg11_bn' --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_T1mean_20210614_SE_block_ratio16_811_V-T_fold2-3_Adam --load-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_T1mean_20210614_SE_block_ratio16_811_V-T_fold2-3_Adam --lr=0.0001 --cuda-idx=2 --data-dropout=False --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --DeepC-normalization-method=WBtop10PercentMean --T1-normalization-method=mean

python test.py --arch='vgg11_bn' --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_T1mean_20210627_SE_block_ratio16_811_V-T_fold2-3_Adam --load-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_T1mean_20210627_SE_block_ratio16_811_V-T_fold2-3_Adam --lr=0.0001 --cuda-idx=2 --data-dropout=False --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --DeepC-normalization-method=WBtop10PercentMean --T1-normalization-method=mean

python test.py --arch='vgg11_bn' --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_T1mean_20210621_SE_block_ratio16_811_V-T_fold3-9_Adam --load-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_T1mean_20210621_SE_block_ratio16_811_V-T_fold3-9_Adam --lr=0.0001 --cuda-idx=1 --data-dropout=False --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --DeepC-normalization-method=WBtop10PercentMean --T1-normalization-method=mean

python test.py --arch='vgg11_bn' --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_T1mean_20210627_SE_block_ratio16_811_V-T_fold9-10_Adam --load-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_T1mean_20210627_SE_block_ratio16_811_V-T_fold9-10_Adam --lr=0.0001 --cuda-idx=2 --data-dropout=False --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --DeepC-normalization-method=WBtop10PercentMean --T1-normalization-method=mean

##811, T1 and ACBV , DSX2,  2021-05-28   SE_block
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx24_5e-5_T1mean_20210719_SE_block_ratio16_811_V-T_fold2-9_Adam --arch='vgg11_bn' --batch-size=24 --lr=5e-5 --cuda-idx=2 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold2' --test-folder='fold9' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx24_5e-5_T1mean_20210719_SE_block_ratio16_811_V-T_fold2-9_Adam

python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_5e-5_T1mean_20210630_SE_block_ratio16_811_V-T_fold2-9_Adam_test24G_GPU_cuda1 --arch='vgg11_bn' --batch-size=5 --lr=5e-5 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold2' --test-folder='fold9' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_5e-5_T1mean_20210630_SE_block_ratio16_811_V-T_fold2-9_Adam_test24G_GPU_cuda1

python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_5e-5_T1mean_20210630_SE_block_ratio16_811_V-T_fold2-9_Adam_test24G_GPU_cuda2 --arch='vgg11_bn' --batch-size=5 --lr=5e-5 --cuda-idx=2 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold2' --test-folder='fold9' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_5e-5_T1mean_20210630_SE_block_ratio16_811_V-T_fold2-9_Adam_test24G_GPU_cuda2

python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_T1mean_20210630_SE_block_ratio16_811_V-T_fold3-9_Adam_test24G_GPU_cuda2 --arch='vgg11_bn' --batch-size=5 --lr=1e-4 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold3' --test-folder='fold9' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_Batchx5_T1mean_20210630_SE_block_ratio16_811_V-T_fold3-9_Adam_test24G_GPU_cuda2

python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_T1mean_20210630_SE_block_ratio16_811_V-T_fold3-9_Adam --arch='vgg11_bn' --batch-size=5 --lr=1e-4 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold3' --test-folder='fold9' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_Batchx5_T1mean_20210630_SE_block_ratio16_811_V-T_fold3-9_Adam

python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_T1mean_20210630_SE_block_ratio16_811_V-T_fold10-9_Adam --arch='vgg11_bn' --batch-size=5 --lr=1e-4 --cuda-idx=2 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold10' --test-folder='fold9' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_Batchx5_T1mean_20210630_SE_block_ratio16_811_V-T_fold10-9_Adam


python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_T1mean_20210630_SE_block_ratio16_811_V-T_fold2-3_Adam --arch='vgg11_bn' --batch-size=5 --lr=1e-4 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold2' --test-folder='fold3' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_Batchx5_T1mean_20210630_SE_block_ratio16_811_V-T_fold2-3_Adam


python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_T1mean_20210630_SE_block_ratio16_811_V-T_fold9-3_Adam --arch='vgg11_bn' --batch-size=5 --lr=1e-4 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold9' --test-folder='fold3' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_Batchx5_T1mean_20210630_SE_block_ratio16_811_V-T_fold9-3_Adam

python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_T1mean_20210630_SE_block_ratio16_811_V-T_fold2-9_Adam --arch='vgg11_bn' --batch-size=5 --lr=1e-4 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold2' --test-folder='fold9' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_Batchx5_T1mean_20210630_SE_block_ratio16_811_V-T_fold2-9_Adam

python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_T1mean_20210630_SE_block_ratio16_811_V-T_fold5-7_Adam --arch='vgg11_bn' --batch-size=5 --lr=1e-4 --cuda-idx=2 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold5' --test-folder='fold7' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_Batchx5_T1mean_20210630_SE_block_ratio16_811_V-T_fold5-7_Adam



python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_T1mean_20210627_SE_block_ratio16_811_V-T_fold9-10_Adam --arch='vgg11_bn' --batch-size=5 --lr=1e-4 --cuda-idx=2 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold9' --test-folder='fold10' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_Batchx5_T1mean_20210627_SE_block_ratio16_811_V-T_fold9-10_Adam


python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_T1mean_20210627_SE_block_ratio16_811_V-T_fold2-3_Adam --arch='vgg11_bn' --batch-size=5 --lr=1e-4 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold2' --test-folder='fold3' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_Batchx5_T1mean_20210627_SE_block_ratio16_811_V-T_fold2-3_Adam



python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_T1mean_20210621_SE_block_ratio16_811_V-T_fold3-9_Adam --arch='vgg11_bn' --batch-size=5 --lr=1e-4 --cuda-idx=2 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold3' --test-folder='fold9' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_Batchx5_T1mean_20210621_SE_block_ratio16_811_V-T_fold3-9_Adam


python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_T1mean_20210614_SE_block_ratio4_811_Adam --arch='vgg11_bn' --batch-size=5 --lr=1e-4 --cuda-idx=2 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold2' --test-folder='fold3' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_Batchx5_T1mean_20210614_SE_block_ratio4_811_Adam

python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_T1mean_20210614_SE_block_ratio8_811_Adam --arch='vgg11_bn' --batch-size=5 --lr=1e-4 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold2' --test-folder='fold3' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_Batchx5_T1mean_20210614_SE_block_ratio8_811_Adam

python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_WB_3T_Batchx5_T1mean_20210614_SE_block_ratio16_811_Adam --arch='vgg11_bn' --batch-size=5 --lr=1e-4 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold2' --test-folder='fold3' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_Batchx5_T1mean_20210614_SE_block_ratio16_811_Adam



### 10-fold validation, DSx2,T1 encoding fixed 2021-04-13  no attention

# others
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_ACBVtop10_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210506_RAW_fold9 --resume=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_ACBVtop10_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_fold9/checkpoint_best.tar --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold9' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_ACBVTop10_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210506_RAW_fold9


###### DeepC only, CUres.
python main.py --save-dir=./result/model/save_vgg11_bn_DeepC_CUiso1mm_3T_DSx2_Top10_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210112_6pm --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=False --input-DeepC=True --DeepC-isotropic=True --DeepC-normalization-method=WBtop10PercentMean |& tee -a ./result/log/log_vgg11_bn_DeepC_CUiso1mm_3T_DSx2_Top10_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210112_6pm

python test.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_DeepC_CUiso1mm_3T_DSx2_Top10_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210112_6pm --load-dir=./result/model/save_vgg11_bn_DeepC_CUiso1mm_3T_DSx2_Top10_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210112_6pm --arch='vgg11_bn' --lr=0.0001 --cuda-idx=2 --input-T1=False --input-DeepC=True --DeepC-isotropic=True --DeepC-normalization-method=WBtop10PercentMean
########   CVPR_CNT4V2T2_SchT4V2T2



###### DeepC only, iso1mm.
python main.py --save-dir=./result/model/save_vgg11_bn_DeepC_CUiso1mm_3T_DSx2_mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210120_7pm --arch='vgg11_bn' --lr=0.0001 --cuda-idx=2 --input-T1=False --input-DeepC=True --DeepC-isotropic=True --DeepC-normalization-method=mean |& tee -a ./result/log/log_vgg11_bn_DeepC_CUiso1mm_3T_DSx2_mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210120_7pm

python test.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_DeepC_CUiso1mm_3T_DSx2_mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210120_7pm --load-dir=./result/model/save_vgg11_bn_DeepC_CUiso1mm_3T_DSx2_mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210120_7pm --arch='vgg11_bn' --lr=0.0001 --cuda-idx=2 --input-T1=False --input-DeepC=True --DeepC-isotropic=True --DeepC-normalization-method=mean
########   CVPR_CNT4V2T2_SchT4V2T2



###### double encoder
#vgg11_bn iso1mm T1 affine and iso1mm DeepC
python main.py --save-dir=./result/model/save_vgg11_bn_T1_affineISO_DeepC_CUdiffeoISO_DSx4_doubleVGG_DiffWeights_T1maxDeepCTop10_lr0d0001_useAUC_20201203_10pm --arch='vgg11_bn' --lr=0.0001 --cuda-idx=2 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean |& tee -a ./result/log/log_vgg11_bn_T1_affineISO_DeepC_CUdiffeoISO_DSx4_doubleVGG_DiffWeights_T1maxDeepCTop10_lr0d0001_useAUC_20201203_10pm

python test.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_affineISO_DeepC_CUdiffeoISO_DSx4_doubleVGG_DiffWeights_T1maxDeepCTop10_lr0d0001_useAUC_20201203_10pm --load-dir=./result/model/save_vgg11_bn_T1_affineISO_DeepC_CUdiffeoISO_DSx4_doubleVGG_DiffWeights_T1maxDeepCTop10_lr0d0001_useAUC_20201203_10pm --arch='vgg11_bn' --lr=0.0001 --cuda-idx=2 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean

#vgg11_bn iso1mm T1 affine and CU resolution for DeepC
python main.py --save-dir=./result/model/save_vgg11_bn_T1_affine_DeepC_diffeoISOcrop_DSx4_doubleVGG_DiffWeights_T1maxDeepCTop10_lr0d0001_useAUC_20201203_4pm --arch='vgg11_bn' --lr=0.0001 --cuda-idx=2 --input_T1_CU_iso1mm_diffeo=True --input-DeepC=True --DeepC-isotropic=False --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean |& tee -a ./result/log/log_vgg11_bn_T1_affine_DeepC_diffeoISOcrop_DSx4_doubleVGG_DiffWeights_T1maxDeepCTop10_lr0d0001_useAUC_20201203_4pm

python test.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_affine_DeepC_diffeoISOcrop_DSx4_doubleVGG_DiffWeights_T1maxDeepCTop10_lr0d0001_useAUC_20201203_4pm --load-dir=./result/model/save_vgg11_bn_T1_affine_DeepC_diffeoISOcrop_DSx4_doubleVGG_DiffWeights_T1maxDeepCTop10_lr0d0001_useAUC_20201203_4pm --arch='vgg11_bn' --lr=0.0001 --cuda-idx=2 --input_T1_CU_iso1mm_diffeo=True --input-DeepC=True --DeepC-isotropic=False --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean


############### 10-fold Cross Validation T-V; double encoder, iso1mm T1 CUaffine and iso1mm DeepC CUaffine
### 10-fold validation, No DS
# fold1
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold1 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold1' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold1

# fold2
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold2 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold2' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold2 

# fold3
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold3 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=2 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold3' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold3 

# fold4
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold4 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold4' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold4 

# fold5
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold5 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold5' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold5

# fold6
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold6 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=2 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold6' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold6

# fold7 
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold7 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold7' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold7

# fold8
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold8 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold8' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold8

# fold9
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold9 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=2 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold9' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold9

# fold10
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold10 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold10' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold10



### 10-fold validation, No DS
# fold1 attention
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold1_Attention --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=mean --val-folder='fold1' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold1_Attention

# fold2 attention
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold2_Attention --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=mean --val-folder='fold2' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold2_Attention

# fold3 attention,2021/04/03
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold3_Attention --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=mean --val-folder='fold3' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold3_Attention 

# fold4 attention
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold4_Attention --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=mean --val-folder='fold4' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold4_Attention

# fold5 attention
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold5_Attention --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=mean --val-folder='fold5' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold5_Attention

# fold6 attention
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold6_Attention --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=mean --val-folder='fold6' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold6_Attention

# fold7 attention
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold7_Attention --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=mean --val-folder='fold7' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold7_Attention

# fold8 attention
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold8_Attention --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=mean --val-folder='fold8' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold8_Attention

# fold9 attention
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold9_Attention --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=mean --val-folder='fold9' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold9_Attention

# fold10 attention
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold10_Attention --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=mean --val-folder='fold10' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold10_Attention


### 10-fold validation, DSx2,T1 encoding fixed 2021-04-13  no attention

# others
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210421_RAW_fold5_T1fixed --arch='vgg11_bn' --lr=0.0001 --cuda-idx=2 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=mean --val-folder='fold5' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210421_RAW_fold5_T1fixed


# fold1 no attention
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210413_RAW_fold1_T1fixed --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=mean --val-folder='fold1' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold1_T1fixed


python test.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold1_T1fixed_only_for_test --load-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold1_T1fixed --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean


# fold2 no attention
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold2_T1fixed --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=mean --val-folder='fold2' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold2_T1fixed

# fold3 no attention,2021/04/03
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold3_T1fixed --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=mean --val-folder='fold3' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold3_T1fixed 

# fold4 no attention
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold4_T1fixed --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=mean --val-folder='fold4' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold4_T1fixed

# fold5 no attention
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold5_T1fixed --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=mean --val-folder='fold5' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold5_T1fixed

# fold6 no attention
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold6_T1fixed --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=mean --val-folder='fold6' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold6_T1fixed

# fold7 no attention
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold7_T1fixed --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=mean --val-folder='fold7' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold7_T1fixed

# fold8 no attention
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold8_T1fixed --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=mean --val-folder='fold8' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold8_T1fixed

# fold9 no attention
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold9_T1fixed --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=mean --val-folder='fold9' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold9_T1fixed

# fold10 no attention
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold10_T1fixed --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=mean --DeepC-normalization-method=mean --val-folder='fold10' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx2_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210403_RAW_fold10_T1fixed


### 10-fold validation, DSx4
# fold1
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx4_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold1 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold1' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx4_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold1

# fold2
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx4_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold2 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold2' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx4_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold2 

# fold3
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx4_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold3 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold3' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx4_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold3 

# fold4
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx4_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold4 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold4' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx4_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold4 

# fold5
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx4_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold5 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold5' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx4_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold5

# fold6
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx4_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold6 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold6' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx4_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold6

# fold7 
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx4_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold7 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold7' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx4_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold7

# fold8
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx4_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold8 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold8' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx4_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold8

# fold9
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx4_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold9 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold9' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx4_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold9

# fold10
python main_10fold.py --save-dir=./result/model/save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx4_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold10 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=0 --input-T1=True --input-DeepC=True --DeepC-isotropic=True --double-vgg=True --double-vgg-share-param=False --T1-normalization-method=max --DeepC-normalization-method=WBtop10PercentMean --val-folder='fold10' |& tee -a ./result/log/log_save_vgg11_bn_T1_ACBV_Affine_WB_3T_T1mean_DSx4_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210327_RAW_fold10



### result_check 
python test.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_MN152Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210217 --load-dir=./result/model/save_vgg11_bn_T1_MN152Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210217 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --data-dropout=False --adaptive-lr=True --input-T1=True --input-DeepC=False --T1-normalization-method=mean

100%|███████████████████████████████████████████| 51/51 [00:17<00:00,  2.94it/s]
 * Accuracy@1 49.020
 * Sensitivity 0.360 Specificity 0.885 AUC 0.549               
ACC raw 0.490 ACC @thr=0.5 0.490 ACC @thr=operating 0.608 ACC max 0.608

python test.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_MN152Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210219 --load-dir=./result/model/save_vgg11_bn_T1_MN152Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210219 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --data-dropout=False --adaptive-lr=True --input-T1=True --input-DeepC=False --T1-normalization-method=mean

  UndefinedMetricWarning)
100%|███████████████████████████████████████████| 51/51 [00:15<00:00,  3.28it/s]
 * Accuracy@1 49.020
 * Sensitivity 0.000 Specificity 1.000 AUC 0.500               
ACC raw 0.490 ACC @thr=0.5 0.490 ACC @thr=operating 0.510 ACC max 0.510

python test.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_MN152Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_MwithDR0d02_20210217 --load-dir=./result/model/save_vgg11_bn_T1_MN152Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_MwithDR0d02_20210217 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --data-dropout=False --adaptive-lr=True --input-T1=True --input-DeepC=False --T1-normalization-method=mean

####





##################### For CAM
python main.py --save-dir=./result/model/save_vgg11_bn_T1_MN152Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210217 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --data-dropout=False --adaptive-lr=True --input-T1=True --input-DeepC=False --T1-normalization-method=mean |& tee -a ./result/log/log_save_vgg11_bn_T1_MN152Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210217

python test.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_MN152Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210217 --load-dir=./result/model/save_vgg11_bn_T1_MN152Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210217 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --data-dropout=False --adaptive-lr=True --input-T1=True --input-DeepC=False --T1-normalization-method=mean

### CAM generation
# Note: change the data path in data_loader T1 and the data path in activation_map.py 
python T1_WB_generate_activation_map_all.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_MN152Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210217 --load-dir=./result/model/save_vgg11_bn_T1_MN152Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210217 --lr=0.0001 --cuda-idx=2 --input-T1=True --input-DeepC=False --T1-normalization-method=mean --channel=T1 



########## 811 CNT4V2T2_SchT4V2T2 DS X 2, 1:32 pm 2/20/21
Are we using T1 as input? :  True
Are we using DeepC as input? :  False
Are we using isotropic DeepC instead of DeepC in CUres (if we use DeepC)? :  False
Are we cropping the isotropic DeepC (if we use isotropic DeepC)? :  False
Are we using double VGG instead of double channel, in case we have 2 inputs? :  True
Not having both T1 and DeepC as input. Argument double_vgg is meaningless. Setting it to False.
For double VGG, do we share the encoding layer parameters? :  True
double_vgg_share_param = True and double_vgg = False. Incompatible. Setting double_vgg_share_param to False.
We will be loading from this directory:  ./result/model/save_vgg11_bn_T1_MN152Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210217
***************
MRIDataset
T1 path: ./dataset/test/ CU_diffeo_WB_iso1mm
Load T1. Total T1 train number is: 51
=> loading checkpoint './result/model/save_vgg11_bn_T1_MN152Affine_WB_3T_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_20210217/checkpoint_best.tar'
=> loaded checkpoint (epoch 54)
  0%|                                                                           | 0/51 [00:00<?, ?it/s]/home/raphael/anaconda3/envs/HumanCBV/lib/python3.7/site-packages/sklearn/metrics/_ranking.py:788: UndefinedMetricWarning: No positive samples in y_true, true positive value should be meaningless
  UndefinedMetricWarning)
100%|██████████████████████████████████████████████████████████████████| 51/51 [00:50<00:00,  1.01it/s]
 * Accuracy@1 90.196
 * Sensitivity 0.960 Specificity 0.846 AUC 0.962               
ACC raw 0.902 ACC @thr=0.5 0.902 ACC @thr=operating 0.882 ACC max 0.902





# CU_diffeo_WB_iso1mm
python main.py --save-dir=./result/model/save_vgg11_bn_T1_CUdiffeo_WB_3T_DSx2_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_MwithDR0d02_20210205 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=2 --data-dropout=False --adaptive-lr=True --input-T1=True --input-DeepC=False --T1-normalization-method=mean |& tee -a ./result/log/log_vgg11_bn_T1_CUdiffeo_WB_3T_DSx2_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_MwithDR0d02_20210205

python test.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_CUdiffeo_WB_3T_DSx2_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_MwithDR0d02_20210205 --load-dir=./result/model/save_vgg11_bn_T1_CUdiffeo_WB_3T_DSx2_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_MwithDR0d02_20210205 --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --data-dropout=False --adaptive-lr=True --input-T1=True --input-DeepC=False --T1-normalization-method=mean
############### Data cite number T-V-T: 4-2-2
100%|████████████████████████████████████████████████████████████████████| 51/51 [00:32<00:00,  1.58it/s]
 * Accuracy@1 86.275
 * Sensitivity 0.960 Specificity 0.769 AUC 0.891               
ACC raw 0.863 ACC @thr=0.5 0.863 ACC @thr=operating 0.843 ACC max 0.863




# T1 affine only
python main.py --save-dir=./result/model/save_vgg11_bn_T1_MN152Affine_WH_3T_DSx2_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_MwithDR0d02_20210105_2am --arch='vgg11_bn' --lr=0.0001 --cuda-idx=2 --data-dropout=False --adaptive-lr=True --input-T1=True --input-DeepC=False --T1-normalization-method=mean |& tee -a ./result/log/log_vgg11_bn_T1_MN152Affine_WH_3T_DSx2_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_MwithDR0d02_20210105_2am

python test.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_MN152Affine_WH_3T_DSx2_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_MwithDR0d02_20210105_2am --load-dir=./result/model/save_vgg11_bn_T1_MN152Affine_WH_3T_DSx2_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_MwithDR0d02_20210105_2am --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --data-dropout=False --adaptive-lr=True --input-T1=True --input-DeepC=False --T1-normalization-method=mean
########  CNT4V2T2_SchT4V2T2
100%|████████████████| 72/72 [00:29<00:00,  2.47it/s]
 * Accuracy@1 79.167
 * Sensitivity 0.657 Specificity 0.973 AUC 0.883               
ACC raw 0.792 ACC @thr=0.5 0.792 ACC @thr=operating 0.806 ACC max 0.819
########


python main.py --save-dir=./result/model/save_vgg11_bn_T1_CUAffine_WB_DHW_3T_xyDSx4_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210107_3pm --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --data-dropout=False --adaptive-lr=True --input-T1=True --input-DeepC=False --T1-normalization-method=mean |& tee -a ./result/log/log_vgg11_bn_T1_CUAffine_WB_DHW_3T_xyDSx4_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210107_3pm

python test.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_CUAffine_WB_DHW_3T_xyDSx4_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210107_3pm --load-dir=./result/model/save_vgg11_bn_T1_CUAffine_WB_DHW_3T_xyDSx4_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210107_3pm --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --data-dropout=False --adaptive-lr=True --input-T1=True --input-DeepC=False --T1-normalization-method=mean
########   CVPR_CNT4V2T2_SchT4V2T2
100%|████████████████████████████████████████████████████| 72/72 [00:10<00:00,  6.59it/s]
 * Accuracy@1 63.889
 * Sensitivity 0.914 Specificity 0.649 AUC 0.817               
ACC raw 0.639 ACC @thr=0.5 0.639 ACC @thr=operating 0.764 ACC max 0.778
########

python main.py --save-dir=./result/model/save_vgg11_bn_T1_CUAffine_WB_DHW_3T_xyDSx4_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210107_8pm --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --data-dropout=True --adaptive-lr=True --input-T1=True --input-DeepC=False --T1-normalization-method=mean |& tee -a ./result/log/log_vgg11_bn_T1_CUAffine_WB_DHW_3T_xyDSx4_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210107_8pm

python test.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_CUAffine_WB_DHW_3T_xyDSx4_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210107_8pm --load-dir=./result/model/save_vgg11_bn_T1_CUAffine_WB_DHW_3T_xyDSx4_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210107_8pm --arch='vgg11_bn' --lr=0.0001 --cuda-idx=1 --data-dropout=True --adaptive-lr=True --input-T1=True --input-DeepC=False --T1-normalization-method=mean
########   CVPR_CNT4V2T2_SchT4V2T2



python main.py --save-dir=./result/model/save_vgg19_bn_T1_CUAffine_WB_DHW_3T_xyDSx4_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210107_8pm --arch='vgg19_bn' --lr=0.0001 --cuda-idx=1 --data-dropout=False --adaptive-lr=True --input-T1=True --input-DeepC=False --T1-normalization-method=mean |& tee -a ./result/log/log_vgg19_bn_T1_CUAffine_WB_DHW_3T_xyDSx4_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210107_8pm

python test.py --save-prediction-numpy-dir ./result/prediction/save_vgg19_bn_T1_CUAffine_WB_DHW_3T_xyDSx4_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210107_8pm --load-dir=./result/model/save_vgg19_bn_T1_CUAffine_WB_DHW_3T_xyDSx4_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210107_8pm --arch='vgg19_bn' --lr=0.0001 --cuda-idx=1 --data-dropout=False --adaptive-lr=True --input-T1=True --input-DeepC=False --T1-normalization-method=mean
########   CVPR_CNT4V2T2_SchT4V2T2



###### T1 WB affine DHW only
python main.py --save-dir=./result/model/save_vgg11_bn_T1_MNIAffine_WB_DHW_iso1mm_3T_DSx2_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210118_9pm --arch='vgg11_bn' --lr=0.0001 --cuda-idx=2 --data-dropout=False --adaptive-lr=True --input-T1=True --input-DeepC=False --T1-normalization-method=mean |& tee -a ./result/log/log_vgg11_bn_T1_MNIAffine_WB_DHW_iso1mm_3T_DSx2_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210118_9pm

python test.py --save-prediction-numpy-dir ./result/prediction/save_vgg11_bn_T1_MNIAffine_WB_DHW_iso1mm_3T_DSx2_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210118_9pm --load-dir=./result/model/save_vgg11_bn_T1_MNIAffine_WB_DHW_iso1mm_3T_DSx2_T1mean_feature16-32-64-128_FC2048-512-2_lr0d0001_CVPR_CNT4V2T2_SchT4V2T2_MwithDR0d02_20210118_9pm --arch='vgg11_bn' --lr=0.0001 --cuda-idx=2 --data-dropout=False --adaptive-lr=True --input-T1=True --input-DeepC=False --T1-normalization-method=mean
### CVPR_CNT4V2T2_SchT4V2T2
100%|█████████████████████████████████████████| 116/116 [02:07<00:00,  1.10s/it]
 * Accuracy@1 76.724
 * Sensitivity 0.772 Specificity 0.814 AUC 0.883               
ACC raw 0.767 ACC @thr=0.5 0.767 ACC @thr=operating 0.784 ACC max 0.793








