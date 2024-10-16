python script/infer_MambaBCD.py  --dataset 'SYSU' \
                                 --model_type 'baseline_tiny_baseline' \
                                 --test_dataset_path '/home/majiancong/data/SYSU/test' \
                                 --decoder_depths 4 \
                                 --if_visible 'gray' \
                                 --cfg '/home/majiancong/MambaCD/changedetection/configs/vssm1/vssm_base_224.yaml' \
                                 --resume '/home/majiancong/MambaCD/changedetection/changedetection/saved_models/SYSU/baseline_base_sysu_sc_now_adjchannel=128_1727600551.7375455/SYSU_best_F1=83.44.pth' \