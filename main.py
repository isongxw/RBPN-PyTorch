from __future__ import print_function
import argparse
from math import log10

import utils
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from rbpn import Net as RBPN
from data import get_training_set, get_eval_set
import pdb
import socket
import time
from tqdm import tqdm
import logging

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4,
                    help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=4,
                    help='training batch size')
parser.add_argument('--testBatchSize', type=int,
                    default=5, help='testing batch size')
parser.add_argument('--start_epoch', type=int, default=1,
                    help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=150,
                    help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=10, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=8,
                    help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed to use. Default=123')
# parser.add_argument('--gpus', default=2, type=int, help='number of gpu')
parser.add_argument('--HR_dir', type=str,
                    default='../datasets/ntire21/train/train_sharp/')
parser.add_argument('--LR_dir', type=str,
                    default='../datasets/ntire21/train/train_sharp_bicubic/X4')
parser.add_argument('--file_list', type=str, default='NTIRE21/train.txt')
parser.add_argument('--other_dataset', type=bool, default=True,
                    help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=bool,
                    default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=5)
parser.add_argument('--patch_size', type=int, default=64,
                    help='0 to use original frame size')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--model_type', type=str, default='RBPN')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--pretrained_sr', default='3x_dl10VDBPNF7_epoch_84.pth',
                    help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights/',
                    help='Location to save checkpoint models')
parser.add_argument('--prefix', default='F5',
                    help='Location to save checkpoint models')

opt = parser.parse_args()
# gpus_list = range(opt.gpus)
gpus_list = [1, 3]
hostname = str(socket.gethostname())
cudnn.benchmark = True

utils.setup_logger('base', 'log/train', 'train_' + opt.prefix,
                   level=logging.INFO, screen=True, tofile=True)
logger = logging.getLogger('base')

logger.info(opt)
logger.info("GPUs: {}, Frames: {}, PatchSize: {}".format(str(gpus_list), str(opt.nFrames), str(opt.patch_size)))


def train(epoch):
    epoch_loss = 0
    model.train()
    with tqdm(training_data_loader, desc="Epoch {} ".format(epoch), ncols=120) as t:
        for iteration, batch in enumerate(t, 1):
            input, target, neigbor, flow = batch[0], batch[1], batch[2], batch[3]
            if cuda:
                input = Variable(input).cuda(gpus_list[0])
                target = Variable(target).cuda(gpus_list[0])
                # bicubic = Variable(bicubic).cuda(gpus_list[0])
                neigbor = [Variable(j).cuda(gpus_list[0]) for j in neigbor]
                flow = [Variable(j).cuda(gpus_list[0]).float() for j in flow]

            optimizer.zero_grad()
            # t0 = time.time()
            prediction = model(input, neigbor, flow)

            # if opt.residual:
            #     prediction = prediction + bicubic

            loss = criterion(prediction, target)
            # t1 = time.time()
            epoch_loss += loss.data
            loss.backward()
            optimizer.step()

            t.set_postfix(Loss=loss.item())

    logger.info("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(
        epoch, epoch_loss / len(training_data_loader)))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    logger.info(net)
    logger.info('Total number of parameters: %d' % num_params)


def checkpoint(epoch):
    ck_name = opt.model_type + "_Epoch_{}.pth".format(epoch)
    ck_path = os.path.join(opt.save_folder, opt.prefix)

    if not os.path.exists(ck_path):
        os.makedirs(ck_path)
    
    model_out_path = os.path.join(ck_path, ck_name)
    torch.save(model.state_dict(), model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))


cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

logger.info('===> Loading datasets')
train_set = get_training_set(opt.LR_dir, opt.HR_dir, opt.nFrames, opt.upscale_factor, opt.data_augmentation,
                             opt.file_list, opt.other_dataset, opt.patch_size, opt.future_frame)
training_data_loader = DataLoader(
    dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

logger.info('===> Building model {}'.format(opt.model_type))
if opt.model_type == 'RBPN':
    model = RBPN(num_channels=3, base_filter=256,  feat=64, num_stages=3,
                 n_resblock=5, nFrames=opt.nFrames, scale_factor=opt.upscale_factor)

model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion = nn.L1Loss()

logger.info('---------- Networks architecture -------------')
print_network(model)
logger.info('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(
            model_name, map_location=lambda storage, loc: storage))
        logger.info('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])
    criterion = criterion.cuda(gpus_list[0])

optimizer = optim.Adam(model.parameters(), lr=opt.lr,
                       betas=(0.9, 0.999), eps=1e-8)

for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    train(epoch)

    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch+1) % (opt.nEpochs/2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        logger.info('Learning rate decay: lr={}'.format(
            optimizer.param_groups[0]['lr']))

    if (epoch+1) % (opt.snapshots) == 0:
        checkpoint(epoch)

utils.email_notification("isongxw@foxmail.com", "Train Finished", "RBPN")
