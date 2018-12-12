#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function
import argparse
import copy
from cnn_finetune import make_model
import click
import cv2
import numpy as np
import torch
from torch.autograd import Variable
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from cnn_finetune import make_model
from grad_cam import (BackPropagation, Deconvolution, GradCAM, GuidedBackPropagation)

# if model has LSTM
# torch.backends.cudnn.enabled = False


def save_gradient(filename, data):
    data -= data.min()
    data /= data.max()
    data *= 255.0
    cv2.imwrite(filename, np.uint8(data))


def save_gradcam(filename, gcam, raw_image):
    h, w, _ = raw_image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    gcam = gcam.astype(np.float) + raw_image.astype(np.float)
    gcam = gcam / gcam.max() * 255.0
    cv2.imwrite(filename, np.uint8(gcam))


model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


def main():

    parser = argparse.ArgumentParser(description='cnn_finetune cifar 10 example')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--model-name', type=str, default='resnet50', metavar='M',
                        help='model name (default: resnet50)')
    parser.add_argument('--model-folder', type=str, default='results_moji/model1', metavar='M',
                        help='model folder (default: results_moji/model1)')
    parser.add_argument('--image-path', type=str, metavar='M',
                        help='model folder (default: )')

    args=parser.parse_args()
    model_path = args.model_folder
    image_path = args.image_path

    CONFIG = {
        'resnet152': {
            'target_layer': '_features.7.2.conv3',
            'input_size': 224
        },
        'vgg19': {
            'target_layer': '_features.36',
            'input_size': 224
        },
        'vgg19_bn': {
            'target_layer': 'features.52',
            'input_size': 224
        },
        'inception_v3': {
            'target_layer': '_features.17.branch_pool.conv',
            'input_size': 299
        },
        'densenet201': {
            'target_layer': 'features.denseblock4',
            'input_size': 224
        },
        'alexnet': {
            'target_layer': 'Convolutional_5',
            'input_size': 227
        },
        'vgg16': {
            'target_layer': 'features.30',
            'input_size': 224
        },
        # Add your model
    }.get(args.model_name)


    if ~args.no_cuda:
        current_device = torch.cuda.current_device()
        print('Running on the GPU:', torch.cuda.get_device_name(current_device))
    else:
        print('Running on the CPU')
    device = torch.device('cuda' if ~args.no_cuda and torch.cuda.is_available() else 'cpu')
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    if((args.model_name=='vgg19') or (args.model_name=='vgg16')):
        model = make_model(
            args.model_name,
            pretrained=True,
            num_classes=len(classes),
            dropout_p=0.2,
            input_size=(32,32)
        )
    else:
        model = make_model(
            args.model_name,
            pretrained=True,
            num_classes=len(classes),
            dropout_p=0.2,
        )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    #print(*list(model.named_modules()), sep='\n')
    # Image
    ''' mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            mean=model.original_model_info.mean,
             std=model.original_model_info.std'''
    raw_image = cv2.imread(image_path)[..., ::-1]
    raw_image = cv2.resize(raw_image, (model.original_model_info.input_size[0], ) * 2)
    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=model.original_model_info.mean,
            std=model.original_model_info.std
        )
    ])(raw_image).unsqueeze(0)

    # =========================================================================
    print('Grad-CAM')
    # =========================================================================
    gcam = GradCAM(model=model)
    probs, idx = gcam.forward(image.to(device))

    for i in range(0, 2):
        gcam.backward(idx=idx[i])
        output = gcam.generate(target_layer=CONFIG['target_layer'])

        save_gradcam('results/{}_gcam_{}.png'.format(classes[idx[i]], args.model_name), output, raw_image)
        print('[{:.5f}] {}'.format(probs[i], classes[idx[i]]))


if __name__ == '__main__':
    main()
