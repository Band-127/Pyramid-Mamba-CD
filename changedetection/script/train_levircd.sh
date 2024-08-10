python script/train_MambaBCD.py --dataset 'LEVIR-CD' \
                                --batch_size 8 \
                                --crop_size 256 \
                                --max_iters 800000 \
                                --model_type baseline_small_4 \
                                --model_param_path 'changedetection/saved_models' \
                                --train_dataset_path '/home/majiancong/data/LEVIR-CD/train' \
                                --test_dataset_path '/home/majiancong/data/LEVIR-CD/test' \
                                --cfg '/home/majiancong/MambaCD/changedetection/configs/vssm1/vssm_small_224.yaml' \
                                --pretrained_weight_path '/home/majiancong/MambaCD/changedetection/vssm_small_0229_ckpt_epoch_222.pth'