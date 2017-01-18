from __future__ import print_function
import argparse

import numpy as np
import six
from tqdm import tqdm
import time

import chainer
from chainer import cuda
import chainer.links as L
from chainer import optimizers

from model import HumanPartsNet
from debugger import Debugger
from data import MiniBatchLoader

resultdir = "./result/"
X_dir = "./data/img/"
y_dir = "./data/mask/"


def train(model, optimizer, MiniBatchLoader, mean_loss, ac, IoU):
    sum_accuracy, sum_loss, sum_IoU = 0, 0, 0
    model.train = True
    MiniBatchLoader.train = True
    for X, y in tqdm(MiniBatchLoader):
        x = chainer.Variable(xp.asarray(X, dtype=xp.float32), volatile='off')
        t = chainer.Variable(xp.asarray(y.astype(np.int32), dtype=xp.int32), volatile='off')
        # optimizer.weight_decay(0.0001)
        optimizer.update(model, x, t)
        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)
        sum_IoU += float(model.IoU) * len(t.data)
    mean_loss.append(sum_loss / MiniBatchLoader.datasize_train)
    ac.append(sum_accuracy / MiniBatchLoader.datasize_train)
    IoU.append(sum_IoU / MiniBatchLoader.datasize_train)
    print('train mean loss={}, accuracy={}, IoU={}'.format(mean_loss[-1], ac[-1], IoU[-1]))
    return model, optimizer, mean_loss, ac, IoU


def test(model, MiniBatchLoader, mean_loss, ac, IoU):
    sum_accuracy, sum_loss, sum_IoU = 0, 0, 0
    model.train = False
    MiniBatchLoader.train = False
    for X, y in tqdm(MiniBatchLoader):
        x = chainer.Variable(xp.asarray(X, dtype=xp.float32), volatile='on')
        t = chainer.Variable(xp.asarray(y.astype(np.int32), dtype=xp.int32), volatile='on')
        loss = model(x, t)
        sum_loss += float(loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)
        sum_IoU += float(model.IoU) * len(t.data)
    mean_loss.append(sum_loss / MiniBatchLoader.datasize_test)
    ac.append(sum_accuracy / MiniBatchLoader.datasize_test)
    IoU.append(sum_IoU / MiniBatchLoader.datasize_test)
    print('train mean loss={}, accuracy={}, IoU={}'.format(mean_loss[-1], ac[-1], IoU[-1]))
    return model, mean_loss, ac, IoU


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Human parts network')
    parser.add_argument('--batchsize', '-b', default=3, type=int,
                        help='Batch size of training')
    parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='Number of epoch of training')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--logflag', '-l', choices=('on', 'off'),
                        default='on', help='Writing and plotting result flag')
    parser.add_argument('--optimizer', '-o', choices=('adam', 'adagrad', 'sgd'),
                        default='sgd', help='Optimizer algorithm')
    parser.add_argument('--lr', '-r', default=1e-7, type=float,
                        help='Learning rate of used optimizer')
    parser.add_argument('--pretrainedmodel', '-p', default=None,
                        help='Path to pretrained model')
    parser.add_argument('--saveflag', '-s', choices=('on', 'off'),
                        default='off', help='Save model and optimizer flag')
    args = parser.parse_args()

    # model setteing
    model = HumanPartsNet(n_class=2)
    if args.pretrainedmodel is not None:
        from chainer import serializers
        serializers.load_hdf5(args.pretrainedmodel, model)
        
    # GPU settings
    if args.gpu >= 0:
        cuda.check_cuda_available()
        xp = cuda.cupy
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    else:
        xp = np

    # Setup optimizer
    # optimizer = optimizers.MomentumSGD(lr=args.lr, momentum=0.99)
    optimizer = optimizers.Adam(alpha=args.lr)
    optimizer.setup(model)

    # prepare data feeder
    MiniBatchLoader = MiniBatchLoader(X_dir, y_dir, batchsize=args.batchsize, insize=model.insize, train=True)
    MiniBatchLoader.scan_for_human()
    debugger = Debugger()

    # error checking enabled
    # chainer.set_debug(True)

    # Learning loop
    train_IoU, test_IoU, train_ac, test_ac, train_mean_loss, test_mean_loss = [], [], [], [], [], []
    stime = time.clock()
    for epoch in six.moves.range(1, args.epoch + 1):
        print('Epoch', epoch, ': training...')
        model, optimizer, train_mean_loss, train_ac, train_IoU = train(model, optimizer, MiniBatchLoader, train_mean_loss, train_ac, train_IoU)
        print('Epoch', epoch, ': testing...')
        model, test_mean_loss, test_ac, test_IoU = test(model, MiniBatchLoader, test_mean_loss, test_ac, test_IoU)

        if args.logflag == 'on':
            etime = time.clock()
            debugger.writelog(MiniBatchLoader.datasize_train, MiniBatchLoader.datasize_test, MiniBatchLoader.batchsize,
                              'Human part segmentation', stime, etime,
                              train_mean_loss, train_ac, train_IoU, 
                              test_mean_loss, test_ac, test_IoU, 
                              epoch, LOG_FILENAME=resultdir + 'log.txt')
            debugger.plot_result(train_mean_loss, test_mean_loss, savename=resultdir + 'log.png')
        if args.saveflag == 'on' and epoch % 10 == 0:
            from chainer import serializers
            serializers.save_hdf5(resultdir + 'humanpartsnet_epoch' + str(epoch) + '.model', model)
            serializers.save_hdf5(resultdir + 'humanpartsnet_epoch' + str(epoch) + '.state', optimizer)
