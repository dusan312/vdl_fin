"""CIFAR10 example for cnn_finetune.
Based on:
- https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py
- https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import argparse
import math
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from cnn_finetune import make_model


parser = argparse.ArgumentParser(description='cnn_finetune cifar 10 example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-name', type=str, default='resnet50', metavar='M',
                    help='model name (default: resnet50)')
parser.add_argument('--dropout-p', type=float, default=0.2, metavar='D',
                    help='Dropout probability (default: 0.2)')
parser.add_argument('--kernel', type=int, default=3, metavar='D',
                    help='Size of kernel (default: 3)')
parser.add_argument('--sigma', type=int, default=2, metavar='D',
                    help='Sigma in normal dist (default: 2)')
parser.add_argument('--experiment-name', type=str, default='defaultname', metavar='M',
                    help='Name (default: defaultname)')


args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


classes = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

if((args.model_name=='vgg19') or (args.model_name=='vgg16')):
    model = make_model(
        args.model_name,
        pretrained=True,
        num_classes=len(classes),
        dropout_p=args.dropout_p,
        input_size=(32,32)
    )
else:
    model = make_model(
        args.model_name,
        pretrained=True,
        num_classes=len(classes),
        dropout_p=args.dropout_p,
    )
model = model.to(device)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=model.original_model_info.mean,
        std=model.original_model_info.std),
])

train_set = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True, num_workers=2
)
test_set = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_set, args.test_batch_size, shuffle=False, num_workers=2
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) / \
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def train(epoch,f):
    total_loss = 0
    total_size = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        filt = get_gaussian_kernel(args.kernel, args.sigma)
        kk = args.kernel-1
        data = F.pad(data, (kk, kk, kk, kk), mode='reflect')
        data = filt(data)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        total_size += data.size(0)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            f.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), total_loss / total_size))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), total_loss / total_size))


def test(f):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    f.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

f = open('./results_moji/log_'+str(args.experiment_name)+'.txt', 'x')
PATH='results_moji/'+str(args.experiment_name)+'_model_t_'+str(args.model_name)+'_'+str(args.epochs)+'_K='+str(args.kernel)+'_S='+str(args.sigma)+'.pth'
acc_best = 0
f.write(str(args.experiment_name) + ': ' + str(args.model_name)+' K='+str(args.kernel)+' S='+str(args.sigma)+str(args))
for epoch in range(1, args.epochs + 1):
    train(epoch,f)
    acc = test(f)
    if acc > acc_best:
        acc_best = acc
        torch.save(model.state_dict(), PATH)

torch.save(model.state_dict(), PATH)
f.close()
