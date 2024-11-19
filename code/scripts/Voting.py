import argparse
import datetime
import os
from pathlib import Path
import time
import shutil
import yaml
import yamlloader
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim
from torch.nn import functional as F
from torch.nn import BCELoss
import pickle

import torch.multiprocessing as mp

from models import UNet_Voting
from datasets import select_dataset_voting
from utils.mylogger import get_mylogger, set_logger_dir
from utils import to_Image, pred_landmarks, visualize, make_dir, voting
from evaluation.eval import Evaluater
from config import get_cfg_defaults, merge_cfg_datasets, merge_cfg_train
from PIL import Image
import numpy as np

class Tester(object):
    def __init__(self, logger, cfg, runs_dir, args=None):
        
        self.cfg = cfg
        self.args = args
        self.runs_dir = runs_dir
        self.evaluater = Evaluater(logger, cfg.dataset.num_landmarks, cfg.dataset.eval_radius)

        self.logger = logger
        self.id_landmarks = [i for i in range(cfg.dataset.num_landmarks)]

        self.visual_dir = os.path.join(self.runs_dir, 'visuals')
        make_dir(self.visual_dir)

        self.uncertainty_dir = os.path.join(self.runs_dir, 'uncertainty')
        make_dir(self.uncertainty_dir)  

        self.pred_dir = os.path.join(self.runs_dir, 'pred')
        make_dir(self.pred_dir)  

        self.visual_dir = os.path.join(self.runs_dir, 'visuals')
        make_dir(self.visual_dir)  

        self.dump_dir = os.path.join(self.runs_dir, 'pseudo_labels')
        make_dir(self.dump_dir)  


    def vote(self, id):
        tmp = torch.from_numpy(np.load(os.path.join(self.temp_dir, f'{id}.npy')))
        heatmap, regression_y, regression_x = tmp[:1], tmp[1:2], tmp[2:3]

        pred_landmark, score_map = voting(heatmap, regression_y, \
            regression_x, self.Radius, multi_process=False, analysis=True)
        
        voting_uncert = score_map.max(-1).max(-1).tolist()
        landmark_list= self.dataloader_1.dataset.get_landmark_gt(id)
            
        with open(os.path.join(self.temp_dir, f'{id}_pred.pkl'), 'wb') as f:
            pickle.dump([pred_landmark, landmark_list], f)
        
        with open(os.path.join(self.pred_dir, f'{id}.pkl'), 'wb') as f:
            pickle.dump([pred_landmark, landmark_list], f)
        
        with open(os.path.join(self.uncertainty_dir, f'{id}.pkl'), 'wb') as f:
            pickle.dump(voting_uncert, f)
        
        with open(os.path.join(self.dump_dir, f'{id}.pkl'), 'wb') as f:
            pickle.dump([pred_landmark, landmark_list], f)
        

    def test(self, net, mode="", dump_label=False):
        self.temp_dir = os.path.join(self.runs_dir, 'temp')
        make_dir(self.temp_dir)

        self.evaluater.reset()

        datset_class = select_dataset_voting(self.cfg.dataset.dataset)
        dataset_1 = datset_class(self.cfg.dataset.dataset_pth, mode, size=cfg.train.input_size, do_repeat=False)
        self.dataloader_1 = DataLoader(dataset_1, batch_size=1,
                                shuffle=False, num_workers=self.cfg.train.num_workers)
        self.Radius = dataset_1.Radius

        img_dict = dict()
        disable_tqdm = True
        for img, _, __, ___, landmark_list, id_str, scale_rate in tqdm(self.dataloader_1, disable=disable_tqdm):
            img = img.cuda()
            id_str = id_str[0]

            with torch.no_grad():
                heatmap, regression_y, regression_x = net(img)

            gray_to_PIL(heatmap[0][3].cpu().detach())\
                .save(os.path.join(self.visual_dir, id_str+'_heatmap.png'))
            
            np.save(os.path.join(self.temp_dir, f'{id_str}.npy'), \
                torch.cat([heatmap, regression_y, regression_x], 0).cpu().numpy())
            
            img_dict[id_str] = img.cpu()
        
        self.logger.info('Inference heatmap and offset map Done')
        
        chosen_id = self.dataloader_1.dataset.list
        process_pool = list()
        for item in chosen_id:
            item = item['ID']
            process = mp.Process(target=self.vote, args=(item,))
            process_pool.append(process)
            process.start()
        
        for process in process_pool:
            process.join()
        self.logger.info('Voting Done')

        for item in chosen_id:
            item = item['ID']
            with open(os.path.join(self.temp_dir, f'{item}_pred.pkl'), 'rb') as f:
                pred_landmark, landmark_list = pickle.load(f)
            self.evaluater.record(pred_landmark, landmark_list, scale_rate)
            self.evaluater.save_img(img_dict[item], pred_landmark, landmark_list, self.runs_dir, item)

        self.evaluater.cal_metrics()
        self.logger.debug('Calculating metrics Done')

        shutil.rmtree(self.temp_dir)
        self.logger.debug('Remove Trash Done')


def L1Loss(pred, gt, mask=None):
    assert (pred.shape == gt.shape)
    gap = pred - gt
    distence = gap.abs()
    if mask is not None:
        distence = distence * mask
    return distence.sum() / mask.sum()

def gray_to_PIL(tensor):
    tensor = tensor  * 255
    images = Image.fromarray(tensor.int().numpy().astype(np.uint8))
    return images

def focal_loss(pred, gt):
    pred = torch.clamp(pred, min=1e-7, max=1 - 1e-7)
    return (-(1 - pred) * gt * torch.log(pred) - pred * (1 - gt) * torch.log(1 - pred)).mean()


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    alpha = 0
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Unet landmark detection network")
    parser.add_argument("--tag", default='femur', help="name of the run")
    parser.add_argument("--dataset", default='femur', choices=['femur'], help="dataset")
    parser.add_argument("--epoch", type=int, default=0, help="default configs")
    parser.add_argument("--test", type=int, default=0, help="default configs")
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    merge_cfg_datasets(cfg, args.dataset)
    merge_cfg_train(cfg, 'voting')

    logger = get_mylogger()
    logger.info(cfg)

    tag = str(datetime.datetime.now()).replace(' ', '-') if args.tag == '' else args.tag
    cfg.tag = tag


    runs_dir = os.path.join("../logs/", tag)
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        runs_path.mkdir()
    cfg.runs_dir = runs_dir
    set_logger_dir(logger, runs_dir)

    net = UNet_Voting(3, cfg.dataset.num_landmarks, non_local=False)
    net = torch.nn.DataParallel(net)
    net = net.cuda()

    ema_net = UNet_Voting(3, cfg.dataset.num_landmarks, non_local=False)
    ema_net = torch.nn.DataParallel(ema_net)
    ema_net = ema_net.cuda()
    for param in ema_net.parameters():
        param.detach_()

    if args.test:
        logger.info("Loading checkpoints")
        checkpoints = torch.load(os.path.join(cfg.runs_dir, "model_epoch.pth"))
        net.load_state_dict(checkpoints)
        tester = Tester(logger, cfg, runs_dir=runs_dir)
        tester.test(net, mode='Test')
        import sys
        sys.exit(0)


    start_epoch = 0
    if args.epoch > 0:
        start_epoch = args.epoch + 1
        logger.info("Loading checkpoints from epoch {}".format(args.epoch))
        checkpoints = torch.load(os.path.join(cfg.runs_dir, \
                            "model_epoch_{}.pth".format(args.epoch)))
        net.load_state_dict(checkpoints)

    optimizer = optim.Adam(params=net.parameters(), \
                           lr=cfg.train.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    scheduler = StepLR(optimizer, cfg.train.decay_step, gamma=cfg.train.decay_gamma)

    loss_logic_fn = focal_loss
    loss_regression_fn = L1Loss
    torch.autograd.set_detect_anomaly(True)

    tester = Tester(logger, cfg, runs_dir=runs_dir)
    tester.test(net, mode='Test')

    if args.test:
        pass

    dataset_class = select_dataset_voting(args.dataset)

    num_iter = 0
    
    for epoch in range(start_epoch, cfg.train.num_epochs):
        logic_loss_list = list()
        net.train()

        dataset = dataset_class(cfg.dataset.dataset_pth, mode='Train', \
                                  size=cfg.train.input_size, pseudo=False)
        dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size,
                    drop_last=True, shuffle=True, num_workers=cfg.train.num_workers)
        
        disable_tqdm = True
        for img, mask, offset_y, offset_x, landmark_list, _, __ in tqdm(dataloader, disable=disable_tqdm):


            img, mask, offset_y, offset_x = img.cuda(), \
                mask.cuda(), offset_y.cuda(), offset_x.cuda()
            
            heatmap, regression_y, regression_x = net(img)

            logic_loss = loss_logic_fn(heatmap, mask)
            regression_loss_y = loss_regression_fn(regression_y, offset_y, mask)
            regression_loss_x = loss_regression_fn(regression_x, offset_x, mask)

            loss = regression_loss_x + regression_loss_y + logic_loss * cfg.train.loss_lambda

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logic_loss_list.append(loss.cpu().item())

            update_ema_variables(net, ema_net, 0.99, num_iter)
            num_iter += 1
            
        logger.info("Epoch {} Training logic loss {} ". \
                    format(epoch, sum(logic_loss_list) / dataset.__len__()))
        scheduler.step()

        if (epoch + 1) % cfg.train.save_seq == 0:
            logger.info(runs_dir + "/model_epoch.pth")
            torch.save(ema_net.state_dict(), runs_dir + "/model_epoch.pth")

            cfg.train.last_epoch = epoch
            ema_net.eval()
            tester.test(ema_net, mode='Test')

        with open(runs_dir + "/config.txt", "w") as f:
            f.write(cfg.dump())

    ema_net.eval()
    tester.test(ema_net, mode='Infer_Train')
