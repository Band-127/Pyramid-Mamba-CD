python script/infer_MambaBCD.py  --dataset 'WHU-CD' \
                                 --model_type 'baseline_whu_nolvl-2layerdecoder' \
                                 --test_dataset_path '/home/majiancong/data/WHU-CD' \
                                 --decoder_depths 4 \
                                 --if_visible 'gray' \
                                 --cfg '/home/majiancong/MambaCD/changedetection/configs/vssm1/vssm_base_224.yaml' \
                                 --resume '/home/majiancong/MambaCD/changedetection/changedetection/saved_models/WHU-CD/baseline_base_whu-nods_1728458521.3596504/77000_model.pth' \