import os
import sys
import time
import math
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import cv2
import torch
import imageio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision.utils import make_grid
from shutil import get_terminal_size
import lpips
import smtplib
from email.mime.text import MIMEText
from email.header import Header


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:

        if not os.path.exists(root):
            os.mkdir(root)

        log_file = os.path.join(
            root, phase + '_{}.log'.format(get_timestamp()))

        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def email_notification(receiver, subject, content):
    '''email notification'''
    mail_host = "smtp.qq.com"  # 设置服务器
    mail_user = "isongxw@foxmail.com"  # 用户名
    mail_pass = "vbpofpiwrtgrghii"  # 口令

    sender = 'isongxw@foxmail.com'

    message = MIMEText(content, 'plain', 'utf-8')
    message['From'] = "isongxw@foxmail.com"
    message['To'] = "isongxw@foxmail.com"

    message['Subject'] = Header(subject, 'utf-8')
    logger = logging.getLogger('base')

    try:
        smtpObj = smtplib.SMTP()
        smtpObj.connect(mail_host, 25)    # 25 为 SMTP 端口号
        smtpObj.login(mail_user, mail_pass)
        smtpObj.sendmail(sender, receiver, message.as_string())
        logger.info("Email notification succeeded")
    except smtplib.SMTPException:
        logger.info("Email notification failed")