import sys
sys.path.append('/home/majiancong/')
import argparse
import os
import time
from thop import profile
import numpy as np
import imageio
from random import randint
from MambaCD.changedetection.configs.config import get_config
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from MambaCD.changedetection.datasets.make_data_loader import ChangeDetectionDatset, make_data_loader
from MambaCD.changedetection.utils_func.metrics import Evaluator
from MambaCD.changedetection.models.MambaPyramid import MambaPyramid

import MambaCD.changedetection.utils_func.lovasz_loss as L
class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.train_data_loader = make_data_loader(args)

        self.evaluator = Evaluator(num_class=2)
        self.writer = SummaryWriter(log_dir=f"./logs/{self.args.model_type}")
        self.deep_model = MambaPyramid(
            pretrained=args.pretrained_weight_path,
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            decoder_depths = args.decoder_depths,
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            drop_rate = args.drop_rate,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            ) 
        
        self.deep_model = self.deep_model.cuda()
        self.model_save_path = os.path.join(args.model_param_path, args.dataset,
                                            args.model_type + '_' + str(time.time()))
        self.lr = args.learning_rate
        self.epoch = args.max_iters // args.batch_size
        torch.random.manual_seed(3407)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        self.optim,               # 优化器
                        T_max=10000,             # 学习率下限
                    )

    def training(self):
        best_kc = 0.0
        best_round = []
        torch.cuda.empty_cache()
        elem_num = len(self.train_data_loader)
        with open(os.path.join(self.model_save_path,'result.txt'),'w') as output:
            output.write(f'best round:{best_round}\n best iter: 0')
        
        train_enumerator = enumerate(self.train_data_loader)
        for _ in tqdm(range(elem_num)):
            itera, data = train_enumerator.__next__()
            pre_change_imgs, post_change_imgs, labels, _ = data

            pre_change_imgs = pre_change_imgs.cuda().float()
            post_change_imgs = post_change_imgs.cuda()
            labels = labels.cuda().long()

            output_1,ds_feature = self.deep_model(pre_change_imgs, post_change_imgs)
            self.optim.zero_grad()
            ce_loss_1 = F.cross_entropy(output_1, labels, ignore_index=255)
            ce_loss_ds = 0
            lovasz_loss = 0
            for index in range(len(ds_feature)):
                ce_loss_ds += F.cross_entropy(ds_feature[index], labels, ignore_index=255)*index/4
                lovasz_loss += L.lovasz_softmax(F.softmax(ds_feature[index], dim=1), labels, ignore=255)*index/4
            lovasz_loss += L.lovasz_softmax(F.softmax(output_1, dim=1), labels, ignore=255)
            main_loss = ce_loss_1 + 0.75 * lovasz_loss + ce_loss_ds
            final_loss = main_loss
            self.writer.add_scalar(tag="ce_loss",scalar_value=ce_loss_1,global_step=itera+1)
            self.writer.add_scalar(tag="ds_loss",scalar_value=ce_loss_ds,global_step=itera+1)
            self.writer.add_scalar(tag="final_loss",scalar_value=final_loss,global_step=itera+1)
            final_loss.backward()
            self.optim.step()
            
            if (itera + 1) % 10 == 0:
                print(f'iter is {itera + 1}, overall loss is {final_loss}')
                if (itera + 1) % 500 == 0:
                    self.deep_model.eval()
                    rec, pre, oa, f1_score, iou, kc = self.validation(iter=itera)
                    if kc > best_kc:
                        torch.save(self.deep_model.state_dict(),
                                   os.path.join(self.model_save_path, f'{itera + 1}_model.pth'))
                        best_iter = itera+1
                        best_kc = kc
                        best_round = [rec, pre, oa, f1_score, iou, kc]
                    print('best round:',best_round)
                    print('best iteration:',best_iter)
                    with open(os.path.join(self.model_save_path,'result.txt'),'w') as output:
                        output.write(f'best round:{best_round}\n best iter: {best_iter}')
                    self.deep_model.train()
        self.writer.close()
        print('The accuracy of the best round is ', best_round)
        print('best iteration:',best_iter)

    def validation(self,iter):
        print('---------starting evaluation-----------')
        self.evaluator.reset()
        dataset = ChangeDetectionDatset(self.args.dataset,self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test')
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=4, drop_last=False)
        torch.cuda.empty_cache()
        with torch.no_grad():
            for itera, data in enumerate(val_data_loader):
                pre_change_imgs, post_change_imgs, labels, name = data             
                pre_change_imgs = pre_change_imgs.cuda().float()
                post_change_imgs = post_change_imgs.cuda().float()
                labels = labels.cuda().long()
                output_1,ds_feature = self.deep_model(pre_change_imgs, post_change_imgs)              
                ce_loss_1 = F.cross_entropy(output_1, labels, ignore_index=255)
                ce_loss_ds = 0
                lovasz_loss = 0
                for index in range(len(ds_feature)):
                    ce_loss_ds += F.cross_entropy(ds_feature[index], labels, ignore_index=255)*index/2
                    lovasz_loss += L.lovasz_softmax(F.softmax(ds_feature[index], dim=1), labels, ignore=255)*index/2
                lovasz_loss += L.lovasz_softmax(F.softmax(output_1, dim=1), labels, ignore=255)
                main_loss = ce_loss_1 + 0.75 * lovasz_loss + ce_loss_ds
                final_loss = main_loss
                self.scheduler.step()
                output_1 = output_1.data.cpu().numpy()
                output_1 = np.argmax(output_1, axis=1)
                labels = labels.cpu().numpy()
                self.evaluator.add_batch(labels, output_1)
                self.writer.add_scalar(tag="ce_loss_validation",scalar_value=ce_loss_1,global_step=iter+1)
                self.writer.add_scalar(tag="ds_loss_validation",scalar_value=ce_loss_ds,global_step=iter+1)
                self.writer.add_scalar(tag="final_loss_validation",scalar_value=final_loss,global_step=iter+1)
        f1_score = self.evaluator.Pixel_F1_score()
        oa = self.evaluator.Pixel_Accuracy()
        rec = self.evaluator.Pixel_Recall_Rate()
        pre = self.evaluator.Pixel_Precision_Rate()
        iou = self.evaluator.Intersection_over_Union()
        kc = self.evaluator.Kappa_coefficient()
        self.writer.add_scalar(tag="f1-score",scalar_value=f1_score,global_step=iter+1)
        self.writer.add_scalar(tag="kc",scalar_value=kc,global_step=iter+1)
        self.writer.add_scalar(tag="IoU",scalar_value=iou,global_step=iter+1)
        print(f'Racall rate is {rec}, Precision rate is {pre}, OA is {oa}, '
              f'F1 score is {f1_score}, IoU is {iou}, Kappa coefficient is {kc}')
        return rec, pre, oa, f1_score, iou, kc


def main():
    parser = argparse.ArgumentParser(description="Training on SYSU/LEVIR-CD/WHU-CD/DSIFN-CD dataset")
    parser.add_argument('--cfg', type=str, default='/home/songjian/project/MambaCD/VMamba/classification/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str)
    parser.add_argument('--dataset', type=str, default='SYSU')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_dataset_path', type=str, default='/home/songjian/project/datasets/SYSU/train')
    # parser.add_argument('--train_data_list_path', type=str, default='/home/songjian/project/datasets/SYSU/train_list.txt')
    parser.add_argument('--test_dataset_path', type=str, default='/home/songjian/project/datasets/SYSU/test')
    parser.add_argument('--decoder_depths', type=int, default=4)
    # parser.add_argument('--test_data_list_path', type=str, default='/home/songjian/project/datasets/SYSU/test_list.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--drop_rate', type=float, default=0.2)
    # parser.add_argument('--train_data_name_list', type=list)
    # parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, default='MambaBCD')
    parser.add_argument('--model_param_path', type=str, default='../saved_models')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=4e-4)

    args = parser.parse_args()

    if args.dataset=='LEVIR-CD' or args.dataset=='LEVIR-CD+':
        args.train_data_name_list = os.listdir(os.path.join(args.train_dataset_path,'A'))
        args.test_data_name_list = os.listdir(os.path.join(args.test_dataset_path,'A'))
    if args.dataset=='SYSU':
        args.train_data_name_list = os.listdir(os.path.join(args.train_dataset_path,'time1'))
        args.test_data_name_list = os.listdir(os.path.join(args.test_dataset_path,'time1'))
    if args.dataset=='DSIFN-CD':
        args.train_data_name_list = os.listdir(os.path.join(args.train_dataset_path,'t1'))
        args.test_data_name_list = os.listdir(os.path.join(args.test_dataset_path,'t1'))
    if args.dataset == 'WHU-CD':
        args.train_data_name_list = []
        args.test_data_name_list = []
        with open('/home/majiancong/WHU-CD/list/test.txt','r') as f_test:
            args.test_data_name_list = f_test.read().split('\n')[:-1]
        with open('/home/majiancong/WHU-CD/list/train.txt','r') as f_test:
            args.train_data_name_list = f_test.read().split('\n')[:-1]

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    main()
