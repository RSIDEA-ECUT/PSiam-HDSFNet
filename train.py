import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import F1Score

import argparse
import logging
import os
import sys
import time
import numpy as np
from segmentation_models_pytorch.utils.meter import AverageValueMeter
from tqdm import tqdm

from datafunc.dataset_load import dataset_load
from tools.seed_tool import seed_everything, seed_workers
from networks.PSiam_HDSFNet import PSiam_HDSFNet


def get_train_loader(args):
    train_set = dataset_load(args.data_dir_train,
                             subset='train',
                             unlabeled=False,
                             train_index=None)
    n_classes = train_set.n_classes
    n_inputs = train_set.n_inputs

    val_size = int(0.2 * len(train_set))
    train_size = len(train_set) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_set, [train_size, val_size])
    train_long = len(train_dataset)
    val_long = len(val_dataset)

    generator = torch.Generator()
    generator.manual_seed(15)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=True,
                              worker_init_fn=seed_workers(rank=0, seed=15),
                              generator=generator)

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=True,
                            worker_init_fn=seed_workers(rank=0, seed=15),
                            generator=generator)

    return train_loader, val_loader, n_inputs, n_classes, train_long, val_long


def opt():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument('--data_dir_train', type=str,
                        default=r'',
                        help='path to training dataset')
    parser.add_argument('--model_path', type=str,
                        default=r'',
                        help='path to save model')
    parser.add_argument('--save', type=str, default='',
                        help='path to save linear classifier')
    parser.add_argument('--save_freq', type=int, default=10, help='number of saving model epochs')
    parser.add_argument('--preview_dir', type=str,
                        default=r'',
                        help='path to preview dir (default: no previews)')
    opt = parser.parse_args()

    opt.save_path = os.path.join(opt.model_path, opt.save)
    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)

    if not os.path.isdir(opt.preview_dir):
        os.makedirs(opt.preview_dir)

    if not os.path.isdir(opt.data_dir_train):
        raise ValueError('data path not exist: {}'.format(opt.data_dir_train))

    return opt


def train():
    args = opt()
    seed_everything(seed=15)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    train_loader, valid_loader, n_inputs, n_classes, train_long, val_long = get_train_loader(args)
    logging.info(f'\tNetwork: PSiam_HDSFNet\n'
                 f'\t\tinput channels: {n_inputs}\n'
                 f'\t\toutput channels (classes): {n_classes}\n'
                 f'\t\tThe size of train dataset: {train_long}\n'
                 f'\t\tThe size of validation dataset: {val_long}\n')

    logging.info(f'''Starting training:
            Epochs:          {args.epochs}
            Batch size:      {args.batch_size}
            Learning rate:   {args.lr}
            Training size:   {len(train_loader)}
            Validation size: {len(valid_loader)}
            Checkpoints:     {args.save}
            Device:          {device.type}
            Images scaling:  {'256'}
        ''')
    model = PSiam_HDSFNet(ms_channels=4, sar_channels=1, n_classes=2).to(device)
    model.initialize_weights()
    if n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss()
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=args.lr)])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    valid_f1 = F1Score(task="binary", num_classes=2).to(device)
    f1 = []

    for epoch in range(args.epochs):
        model.train()
        train_loader = tqdm(train_loader, file=sys.stdout)
        sample_num = 0
        inference_time = 0.0
        loss_meter1 = AverageValueMeter()
        avg_inference_time_meter1 = AverageValueMeter()
        train_loss = 0.0
        for trainstep, traindata in enumerate(train_loader):
            trainimage, traintarget = traindata['image'], traindata['label']
            trainimage = trainimage.to(device)
            mask_type = torch.float32 if n_classes == 1 else torch.int64
            traintarget = traintarget.to(device, dtype=mask_type)
            sample_num += trainimage.shape[0]
            start_time = time.time()
            trainpredict = model(trainimage)
            end_time = time.time()
            inference_time += (end_time - start_time)
            loss = criterion(trainpredict, traintarget)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_value = loss.cpu().detach().numpy()
            loss_meter1.add(loss_value)
            train_loss_logs = {type(criterion).__name__: loss_meter1.mean}
            avg_train_inference_time = inference_time / sample_num
            avg_inference_time_meter1.add(avg_train_inference_time)
            avg_inference_time_logs1 = {"avg_train_inference_time": avg_inference_time_meter1.mean}
            train_loss_logs.update(avg_inference_time_logs1)
            str_logs1 = ["{} - {:.4}".format(k, np.mean(v)) for k, v in train_loss_logs.items()]
            logs1 = [f'Train epoch - {epoch + 1}/{args.epochs}']
            logs1 += str_logs1
            s1 = ", ".join(logs1)
            train_loader.set_postfix_str(s1)

        train_loss = train_loss / len(train_loader)
        logging.info(f'\tThe train_loss of this epoch is {train_loss: .6f}\n')

        model.eval()
        with torch.no_grad():
            valid_loader = tqdm(valid_loader, file=sys.stdout)
            sample_num = 0
            inference_time = 0.0
            loss_meter2 = AverageValueMeter()
            avg_inference_time_meter2 = AverageValueMeter()
            val_loss = 0.0
            for valstep, valdata in enumerate(valid_loader):
                valimage, valtarget = valdata['image'], valdata['label']
                valimage = valimage.to(device)
                mask_type = torch.float32 if n_classes == 1 else torch.int64
                valtarget = valtarget.to(device, dtype=mask_type)
                sample_num += valimage.shape[0]
                start_time = time.time()
                valpredict = model(valimage)
                end_time = time.time()
                inference_time += (end_time - start_time)

                loss = criterion(valpredict, valtarget)
                val_loss += loss.item()
                valid_f1.update(valpredict.argmax(1), valtarget)
                loss_value = loss.cpu().detach().numpy()
                loss_meter2.add(loss_value)
                val_loss_logs = {type(criterion).__name__: loss_meter2.mean}
                avg_val_inference_time = inference_time / sample_num
                avg_inference_time_meter2.add(avg_val_inference_time)
                avg_inference_time_logs2 = {"avg_val_inference_time": avg_inference_time_meter2.mean}
                val_loss_logs.update(avg_inference_time_logs2)
                str_logs2 = ["{} - {:.4}".format(k, np.mean(v)) for k, v in val_loss_logs.items()]
                logs2 = [f'Valid epoch - {epoch + 1}/{args.epochs}']
                logs2 += str_logs2
                s2 = ", ".join(logs2)
                valid_loader.set_postfix_str(s2)
        val_loss = val_loss / len(valid_loader)

        valid_f11 = valid_f1.compute().cpu().detach().numpy()
        f1.append(valid_f11)
        logging.info(f'\t\t[Val] F1: {valid_f11: .4f}\n')
        valid_f1.reset()

        if (epoch + 1) >= 10 and ((epoch + 1) % args.save_freq) == 0:
            print('==> Saving The Epoch...')
            state = {
                'PSiam': model.state_dict()
            }
            save_name = 'ckpt_{model}_{epoch}.pth'.format(model='PSiam-HDSFNet', epoch=epoch + 1)
            save_name = os.path.join(args.save_path, save_name)
            torch.save(state, save_name)
            logging.info(f'Checkpoint {epoch + 1} saved !')

        if f1[-1] >= np.max(f1):
            print('==> Saving The Best...')
            state = {
                'PSiam': model.state_dict()
            }
            save_name = 'ckpt_{model}_{epoch}_{f1}.pth'.format(model='PSiam-HDSFNet', epoch=epoch + 1,
                                                               f1=f1[-1] * 1e4)
            save_name = os.path.join(args.save_path, save_name)
            torch.save(state, save_name)
            logging.info(f'Checkpoint {f1[-1]: .6f} saved !')
        torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
