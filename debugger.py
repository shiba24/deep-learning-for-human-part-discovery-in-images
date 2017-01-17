import numpy as np
import logging
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Debugger(object):
    """Debugger for ImageGenerater dataset"""
    def writelog(self, N_train, N_test, batchsize, netstructure, stime, etime,
                 train_mean_loss, train_ac, train_IoU, 
                 test_mean_loss, test_ac, test_IoU, 
                 epoch, LOG_FILENAME='log.txt'):
        logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG, format='%(asctime)s %(message)s')
        logging.info('New trial **************************************************\n'
                     'All data: %d frames, train: %d frames / test: %d frames.\n'
                     '   Network = %s, Batchsize = %d.\n'
                     '   Total Time = %.3f sec.\n'
                     '   Epoch: 1,  train mean loss=  %.5f, test mean loss=  %.5f.\n'
                     '              train accuracy =  %.5f, test accuracy =  %.5f.\n'
                     '              train IoU      =  %.5f, test IoU      =  %.5f.\n'
                     '   Epoch: %d, train mean loss=  %.5f, test mean loss=  %.5f.\n'
                     '              train accuracy =  %.5f, test accuracy =  %.5f.\n'
                     '              train IoU      =  %.5f, test IoU      =  %.5f.\n',
                     N_train + N_test, N_train, N_test,
                     netstructure, batchsize,
                     etime - stime,
                     train_mean_loss[0], test_mean_loss[0],
                     train_ac[0], test_ac[0],
                     train_IoU[0], test_IoU[0],
                     epoch,
                     train_mean_loss[-1], test_mean_loss[-1],
                     train_ac[-1], test_ac[-1],
                     train_IoU[-1], test_IoU[-1])
        f = open(LOG_FILENAME, 'rt')
        try:
            body = f.read()
        finally:
            f.close()
        print('FILE:')
        print(body)

    def plot_result(self, train_mean_loss, test_mean_loss, savename='result.png'):
        ep = np.arange(len(train_mean_loss)) + 1
        plt.plot(ep, train_mean_loss, color="blue", linewidth=2.5, linestyle="-", label="Train")
        plt.plot(ep, test_mean_loss, color="red",  linewidth=2.5, linestyle="-", label="Test")
        plt.title("Mean Loss")
        plt.xlabel("epoch")
        plt.legend(loc='upper right')
        plt.savefig(savename)
        plt.close()
