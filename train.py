#!/usr/bin/env python
from __future__ import print_function
import argparse

import numpy as np
import six
from tqdm import tqdm

import chainer
from chainer import cuda
import chainer.links as L
from chainer import optimizers


from model import HumanPartsNet

import time


resultdir = "./result/"


from chainer import training
from chainer.training import extensions


model = L.Classifier(MnistModel())
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

train, test = chainer.datasets.get_mnist()

train_iter = chainer.iterators.SerialIterator(train, 100)
test_iter = chainer.iterators.SerialIterator(test, 100, repeat=False, shuffle=False)

updater = training.StandardUpdater(train_iter, optimizer, device=-1)
trainer = training.Trainer(updater, (100, 'epoch'), out="result")

trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
trainer.extend(extensions.ProgressBar())

trainer.run()





parser = argparse.ArgumentParser(description='Human parts network')
parser.add_argument('--augmentation', '-a', default=1.0, type=float,
                    help='The amount of data augmentation')
parser.add_argument('--batchsize', '-b', default=100, type=int,
                    help='Batch size of training')
parser.add_argument('--data', '-d', choices=('on', 'off'),
                    default='off', help='Data normalization & augmentation')
parser.add_argument('--epoch', '-e', default=30, type=int,
                    help='Number of epoch of training')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--logflag', '-l', choices=('on', 'off'),
                    default='on', help='Writing log flag')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--net', '-n', choices=('alex', 'alexbn', 'googlenet'),
                    default='alexbn', help='Network type')
parser.add_argument('--optimizer', '-o', choices=('adam', 'adagrad', 'sgd'),
                    default='sgd', help='Optimizer algorithm')
parser.add_argument('--plotflag', '-p', choices=('on', 'off'),
                    default='off', help='Accuracy plot flag')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--saveflag', '-s', choices=('on', 'off'),
                    default='off', help='Save model and optimizer flag')
args = parser.parse_args()


def train(model, optimizer, MinibatchLoader, mean_loss, ac):
    sum_accuracy, sum_loss = 0, 0
    model.train = True
    MinibatchLoader.train = True
    for X, y in tqdm(MinibatchLoader):
        x = chainer.Variable(xp.asarray(X), volatile='off')
        t = chainer.Variable(xp.asarray(y.reshape(-1)), volatile='off')
        # optimizer.weight_decay(0.0001)
        optimizer.update(model, x, t)
        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)
    print('train mean loss={}, accuracy={}'.format(sum_loss / MinibatchLoader.datasize_train, sum_accuracy / MinibatchLoader.datasize_train))
    mean_loss.append(sum_loss / MinibatchLoader.datasize_train)
    ac.append(sum_accuracy / MinibatchLoader.datasize_train)
    return model, optimizer, mean_loss, ac


def test(model, MinibatchLoader, mean_loss, ac):
    sum_accuracy, sum_loss = 0, 0
    model.train = False
    MinibatchLoader.train = False
    for X, y in tqdm(MinibatchLoader):
        x = chainer.Variable(xp.asarray(X), volatile='on')
        t = chainer.Variable(xp.asarray(y), volatile='on')
        loss = model(x, t)
        sum_loss += float(loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)
    print('test  mean loss={}, accuracy={}'.format(
        sum_loss / MinibatchLoader.datasize_test, sum_accuracy / MinibatchLoader.datasize_test))
    mean_loss.append(sum_loss / MinibatchLoader.datasize_test)
    ac.append(sum_accuracy / MinibatchLoader.datasize_test)
    return model, mean_loss, ac


# model setteing
# Init/Resume
# future 
model = HumanPartsNet()

# GPU settings
if args.gpu >= 0:
    cuda.check_cuda_available()
    xp = cuda.cupy
    cuda.get_device(args.gpu).use()
    model.to_gpu()
else: xp = np

# Setup optimizer
optimizer = optimizers.MomentumSGD(lr=1e-10, momentum=0.99)
optimizer.setup(model)

# prepare data feeder
MinibatchLoader = MinibatchLoader(, args.batchsize, train=True, insize=300)
debugger = Debugger()

# Learning loop
train_ac, test_ac, train_mean_loss, test_mean_loss = [], [], [], []
stime = time.clock()


# Learning loop
for epoch in six.moves.range(1, args.epoch + 1):
    print('epoch', epoch)
    print('Training...')
    model, optimizer, train_mean_loss, train_ac = train(model, optimizer, MinibatchLoader, train_mean_loss, train_ac)
    print('Testing...')
    model, test_mean_loss, test_ac = test(model, MinibatchLoader, test_mean_loss, test_ac)


    if args.logflag == 'on':
        etime = time.clock()
        debugger.writelog(MinibatchLoader.datasize_train, MinibatchLoader.datasize_test, MinibatchLoader.batchsize,
                          'Human part segmentation', stime, etime,
                          train_mean_loss, train_ac, test_mean_loss, test_ac, epoch, LOG_FILENAME=resultdir + 'log.txt')
    if args.plotflag == 'on':
        debugger.plot_result(train_mean_loss, test_mean_loss, savename='log.png')
    if args.saveflag == 'on' and epoch % 10 == 0:
        from chainer import serializers
        serializers.save_hdf5(resultdir + 'humanpartsnet_epoch'+ str(epoch) + '.model', model)
        serializers.save_hdf5(resultdir + 'humanpartsnet_epoch'+ str(epoch) + '.state', optimizer)

