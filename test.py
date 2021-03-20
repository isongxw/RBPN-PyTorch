from __future__ import print_function
import argparse
import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from rbpn import Net as RBPN
from data import get_test_set
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import time
import cv2
import logging
import utils

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4,
                    help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int,
                    default=20, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--chop_forward', type=bool, default=True)
parser.add_argument('--threads', type=int, default=8,
                    help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123,
                    help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--HR_dir', type=str,
                    default='')
parser.add_argument('--LR_dir', type=str,
                    default='../datasets/ntire21/test/test_sharp_bicubic/X4')
parser.add_argument('--file_list', type=str, default='NTIRE21/test.txt')
parser.add_argument('--other_dataset', type=bool, default=True,
                    help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=bool,
                    default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=5)
parser.add_argument('--model_type', type=str, default='RBPN')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--output', default='Final_Results',
                    help='Location to save checkpoint models')
parser.add_argument('--model', default='weights/F5/RBPN_Epoch_149.pth',
                    help='sr pretrained base model')

opt = parser.parse_args()

gpus_list = [0,1]

utils.setup_logger('base', 'log/test', 'test',
                   level=logging.INFO, screen=True, tofile=True)
logger = logging.getLogger('base')

logger.info(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

logger.info('Loading datasets')
test_set = get_test_set(opt.LR_dir, opt.HR_dir, opt.nFrames, opt.upscale_factor,
                        opt.file_list, opt.other_dataset, opt.future_frame)
testing_data_loader = DataLoader(
    dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

logger.info('Building model {}'.format(opt.model_type))
if opt.model_type == 'RBPN':
    model = RBPN(num_channels=3, base_filter=256,  feat=64, num_stages=3,
                 n_resblock=5, nFrames=opt.nFrames, scale_factor=opt.upscale_factor)

if cuda:
    model = torch.nn.DataParallel(model, device_ids=gpus_list)

model.load_state_dict(torch.load(
    opt.model, map_location=lambda storage, loc: storage))
logger.info('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])


def eval():
    model.eval()
    count = 1
    avg_psnr_predicted = 0.0
    avg_ssim_predicted = 0.0
    avg_time_predicted = 0.0
    # batch_size = opt.testBatchSize
    for batch in testing_data_loader:
        input, targets, neigbor, flow, bicubic, filename = batch[
            0], batch[1], batch[2], batch[3], batch[4], batch[5]

        with torch.no_grad():
            input = Variable(input).cuda(gpus_list[0])
            bicubic = Variable(bicubic).cuda(gpus_list[0])
            neigbor = [Variable(j).cuda(gpus_list[0]) for j in neigbor]
            flow = [Variable(j).cuda(gpus_list[0]).float() for j in flow]

        t0 = time.time()
        if opt.chop_forward:
            with torch.no_grad():
                predictions = chop_forward(
                    input, neigbor, flow, model, opt.upscale_factor)
        else:
            with torch.no_grad():
                predictions = model(input, neigbor, flow)

        t1 = time.time()
        time_predicted = (t1 - t0) / len(predictions)
        
        for i in range(len(predictions)):
            cur_video = filename[i].split('/')[-2]
            cur_frame = filename[i].split('/')[-1]
            prediction = utils.tensor2img(predictions[i])

            save_img(prediction, cur_video + '_' + cur_frame, False)
            avg_time_predicted += time_predicted
            count += 1

            if opt.HR_dir != '':
                target = utils.tensor2img(targets[i])
                psnr_predicted = PSNR(target, prediction)
                ssim_predicted = SSIM(target, prediction)
                
                avg_psnr_predicted += psnr_predicted
                avg_ssim_predicted += ssim_predicted
                
                logger.info("Processing: %s || PSNR: %.4f || SSIM: %.4f || Timer: %.4f sec." %
                            (cur_video + '_' + cur_frame, psnr_predicted, ssim_predicted, time_predicted))
            else:
                logger.info("Processing: %s || Timer: %.4f sec." % (cur_video + '_' + cur_frame, time_predicted))
    if opt.HR_dir != '':
        logger.info("AVG_PSNR = {}".format(avg_psnr_predicted / count))
        logger.info("AVG_SSIM = {}".format(avg_ssim_predicted / count))
    
    logger.info("AVG_TIME = {}".format(avg_time_predicted / count))


def save_img(img, img_name, pred_flag):
    # save img
    save_dir = opt.output
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_fn = save_dir + '/' + img_name
    cv2.imwrite(save_fn, img)


def PSNR(res_img, ref_img):
    psnr = peak_signal_noise_ratio(ref_img, res_img)
    return psnr


def SSIM(res_img, ref_img):
    ssim = structural_similarity(
        ref_img, res_img, multichannel=True, gaussian_weights=True, use_sample_covariance=False)
    return ssim


def chop_forward(x, neigbor, flow, model, scale, shave=8, min_size=2000, nGPUs=opt.gpus):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        [x[:, :, 0:h_size, 0:w_size], [j[:, :, 0:h_size, 0:w_size]
                                       for j in neigbor], [j[:, :, 0:h_size, 0:w_size] for j in flow]],
        [x[:, :, 0:h_size, (w - w_size):w], [j[:, :, 0:h_size, (w - w_size):w]
                                             for j in neigbor], [j[:, :, 0:h_size, (w - w_size):w] for j in flow]],
        [x[:, :, (h - h_size):h, 0:w_size], [j[:, :, (h - h_size):h, 0:w_size]
                                             for j in neigbor], [j[:, :, (h - h_size):h, 0:w_size] for j in flow]],
        [x[:, :, (h - h_size):h, (w - w_size):w], [j[:, :, (h - h_size):h, (w - w_size):w] for j in neigbor], [j[:, :, (h - h_size):h, (w - w_size):w] for j in flow]]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            with torch.no_grad():
                # torch.cat(inputlist[i:(i + nGPUs)], dim=0)
                input_batch = inputlist[i]
                output_batch = model(
                    input_batch[0], input_batch[1], input_batch[2])
            outputlist.extend(output_batch.chunk(nGPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch[0], patch[1], patch[2],
                         model, scale, shave, min_size, nGPUs)
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    with torch.no_grad():
        output = Variable(x.data.new(b, c, h, w))
    output[:, :, 0:h_half, 0:w_half] \
        = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


eval()
