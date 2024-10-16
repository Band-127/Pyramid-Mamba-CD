python script/infer_MambaBCD.py  --dataset 'LEVIR-CD' \
                                 --model_type 'baseline_tiny_baseline' \
                                 --test_dataset_path '/home/majiancong/data/LEVIR-CD/test' \
                                 --decoder_depths 4 \
                                 --if_visible 'gray' \
                                 --cfg '/home/majiancong/MambaCD/changedetection/configs/vssm1/vssm_small_224.yaml' \
                                 --resume '/home/majiancong/MambaCD/changedetection/changedetection/saved_models/LEVIR-CD/baseline_tiny_baseline_1725679764.1621282/45500_model.pth' \