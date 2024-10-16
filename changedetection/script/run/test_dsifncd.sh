python script/infer_MambaBCD.py  --dataset 'DSIFN-CD' \
                                 --model_type 'baseline_base_baseline' \
                                 --test_dataset_path '/home/majiancong/data/DSIFN-CD/test' \
                                 --decoder_depths 4 \
                                 --if_visible 'gray' \
                                 --cfg '/home/majiancong/MambaCD/changedetection/configs/vssm1/vssm_base_224.yaml' \
                                 --resume '/home/majiancong/MambaCD/changedetection/changedetection/saved_models/DSIFN-CD/baseline_base_dsifn_kaiming_init_loss_dropout_valid-consine_1727625713.3368802/71500_model.pth'