python script/infer_MambaBCD.py  --dataset 'LEVIR-CD+' \
                                 --model_type 'MambaBCD_Tiny' \
                                 --test_dataset_path '<dataset_path>/LEVIR-CD+/test' \
                                 --test_data_list_path '<dataset_path>/LEVIR-CD+/test_list.txt' \
                                 --cfg '<project_path>/MambaCD/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml' \
                                 --pretrained_weight_path '<project_path>/MambaCD/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth'\
                                 --resume '<saved_model_path>/MambaBCD_Tiny_LEVIRCD+_F1_0.8803.pth'