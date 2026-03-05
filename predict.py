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
from networks.PSiam_HDSFNet import PSiam_HDSFNet


def get_test_loader(args):
    test_set = dataset_load(args.data_dir_predict,
                            subset='test',
                            unlabeled=False,
                            train_index=None)
    n_classes = test_set.n_classes
    n_inputs = test_set.n_inputs

    test_size = len(test_set)
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             drop_last=False)

    return test_loader, n_inputs, n_classes, test_size


def opt():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--data_dir_predict', type=str,
                        default=r'',
                        help='path to test dataset')
    parser.add_argument('--pretrain_model', '-m',
                        default=r'')
    parser.add_argument('--preview_dir', type=str,
                        default=r'',
                        help='path to preview dir (default: no previews)')
    opt = parser.parse_args()
    if not os.path.isdir(opt.data_dir_predict):
        raise ValueError('data path not exist: {}'.format(opt.data_dir_predict))
    return opt


def predict():
    args = opt()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    predict_loader, n_inputs, n_classes, test_long = get_test_loader(args)
    logging.info(f'\tNetwork: PP2NetV35\n'
                 f'\t\t{n_inputs} input channels\n'
                 f'\t\t{n_classes} output channels (classes)\n')
    logging.info(f'''Starting testing:
            Batch size:      {args.batch_size}
            Learning rate:   {args.lr}
            testing size:   {test_long}
            Checkpoints:     {args.pretrain_model}
            Device:          {device.type}
            Images scaling:  {'256'}
        ''')
    model = PSiam_HDSFNet(ms_channels=4, sar_channels=1, n_classes=2).to(device)
    model.initialize_weights()

    try:
        print('==>loading pretrained Linear model')
        load_params = torch.load(args.pretrain_model,
                                 map_location=args.device)
        model.load_state_dict(load_params['PSiam'])
    except FileNotFoundError:
        print("Pre-trained weights not found. Please to check.")
    model.eval()

    if n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss()

    test_f1 = F1Score(task="binary", num_classes=2, average='none').to(device)

    model.eval()
    with torch.no_grad():
        predict_loader = tqdm(predict_loader, file=sys.stdout)
        sample_num = 0
        inference_time = 0.0
        loss_meter = AverageValueMeter()
        avg_inference_time_meter = AverageValueMeter()

        for teststep, testdata in enumerate(predict_loader):
            test_loss = 0.0
            testimage, testtarget = testdata['image'], testdata['label']
            testimage = testimage.to(device)
            mask_type = torch.float32 if n_classes == 1 else torch.int64
            testtarget = testtarget.to(device, dtype=mask_type)
            sample_num += testimage.shape[0]
            start_time = time.time()
            testpredict = model(testimage)
            end_time = time.time()
            inference_time += (end_time - start_time)

            loss = criterion(testpredict, testtarget)
            test_loss += loss.cpu().detach().numpy()
            test_f1.update(testpredict.argmax(1), testtarget)

            loss_value = loss.cpu().detach().numpy()
            loss_meter.add(loss_value)
            val_loss_logs = {type(criterion).__name__: loss_meter.mean}
            avg_val_inference_time = inference_time / sample_num
            avg_inference_time_meter.add(avg_val_inference_time)
            avg_inference_time_logs2 = {"avg_val_inference_time": avg_inference_time_meter.mean}
            val_loss_logs.update(avg_inference_time_logs2)
            str_logs = ["{} - {:.4}".format(k, np.mean(v)) for k, v in val_loss_logs.items()]
            logs = [f'Valid epoch - {teststep + 1}/{len(predict_loader)}']
            logs += str_logs
            s = ", ".join(logs)
            predict_loader.set_postfix_str(s)

            test_f11 = test_f1.compute().cpu().detach().numpy()
            logging.info(f'\t\t[Test] F1: {test_f11: .4f}\n')
            test_f1.reset()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    predict()
