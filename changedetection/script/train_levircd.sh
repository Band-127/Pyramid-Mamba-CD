python script/train_MambaBCD.py --dataset 'LEVIR-CD' \
                                --batch_size 16 \
                                --crop_size 256 \
                                --max_iters 800000 \
                                --model_type MambaBDA_Tiny \
                                --model_param_path 'changedetection/saved_models' \
                                --train_dataset_path '/home/majiancong/data/LEVIR-CD/train/A' \
                                --test_dataset_path '/home/majiancong/data/LEVIR-CD/test/A' \
                                --cfg '/home/majiancong/MambaCD/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml' \
                                --pretrained_weight_path '/home/majiancong/MambaCD/changedetection/vssm_tiny_0230_ckpt_epoch_262.pth'