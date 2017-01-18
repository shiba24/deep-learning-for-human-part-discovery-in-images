import chainer
import chainer.links as L
import chainer.functions as F
from chainer import serializers
from chainer import cuda
import cPickle as pickle
import numpy as np
import os
import six


url = 'https://googledrive.com/host/0BxSyYt1jT6LhUlhITjdicDFyNHM'
modelname = 'fcn-8s-pascalcontext_W_and_b.pkl'

"""
This URL is for the pre-trained model, VGG 16 with fcn 8s pascal context.

For more details and the LICENSE, please see https://github.com/shiba24/pretrained-model-collections .
"""


class HumanPartsNet(chainer.Chain):
    """
    Human parts Convnets proposed by the paper.

    """
    insize = 300

    def __init__(self, VGGModel=None, n_class=2):
        if VGGModel is None:
            self.wb = load_VGGmodel()
        else:
            self.wb = VGGModel
        self.n_class = n_class
        # layers which is trained
        super(HumanPartsNet, self).__init__(
            conv1_1=L.Convolution2D(  3,  64, 3, stride=1, pad=100, initialW=self.wb["conv1_1_W"], initial_bias=self.wb["conv1_1_b"]),
            conv1_2=L.Convolution2D( 64,  64, 3, stride=1, pad=1, initialW=self.wb["conv1_2_W"], initial_bias=self.wb["conv1_2_b"]),
            conv2_1=L.Convolution2D( 64, 128, 3, stride=1, pad=1, initialW=self.wb["conv2_1_W"], initial_bias=self.wb["conv2_1_b"]),
            conv2_2=L.Convolution2D(128, 128, 3, stride=1, pad=1, initialW=self.wb["conv2_2_W"], initial_bias=self.wb["conv2_2_b"]), 
            conv3_1=L.Convolution2D(128, 256, 3, stride=1, pad=1, initialW=self.wb["conv3_1_W"], initial_bias=self.wb["conv3_1_b"]),
            conv3_2=L.Convolution2D(256, 256, 3, stride=1, pad=1, initialW=self.wb["conv3_2_W"], initial_bias=self.wb["conv3_2_b"]),
            conv3_3=L.Convolution2D(256, 256, 3, stride=1, pad=1, initialW=self.wb["conv3_3_W"], initial_bias=self.wb["conv3_3_b"]),
            conv4_1=L.Convolution2D(256, 512, 3, stride=1, pad=1, initialW=self.wb["conv4_1_W"], initial_bias=self.wb["conv4_1_b"]),
            conv4_2=L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=self.wb["conv4_2_W"], initial_bias=self.wb["conv4_2_b"]),
            conv4_3=L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=self.wb["conv4_3_W"], initial_bias=self.wb["conv4_3_b"]), 
            conv5_1=L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=self.wb["conv5_1_W"], initial_bias=self.wb["conv5_1_b"]),
            conv5_2=L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=self.wb["conv5_2_W"], initial_bias=self.wb["conv5_2_b"]),
            conv5_3=L.Convolution2D(512, 512, 3, stride=1, pad=1, initialW=self.wb["conv5_3_W"], initial_bias=self.wb["conv5_3_b"]),
 
            upsample_pool1=L.Convolution2D(64, self.n_class, ksize=1, stride=1, pad=0, wscale=0.01),
            upsample_pool2=L.Convolution2D(128, self.n_class, ksize=1, stride=1, pad=0, wscale=0.01),
            upsample_pool3=L.Convolution2D(256, self.n_class, ksize=1, stride=1, pad=0, wscale=0.01),
            upsample_pool4=L.Convolution2D(512, self.n_class, ksize=1, stride=1, pad=0, wscale=0.01),

            fc6_conv=L.Convolution2D(512, 4096, 7, stride=1, pad=0, initialW=self.wb["fc6_W"], initial_bias=self.wb["fc6_b"]),
            fc7_conv=L.Convolution2D(4096, 4096, 1, stride=1, pad=0, initialW=self.wb["fc7_W"], initial_bias=self.wb["fc7_b"]),

            upconv1=L.Deconvolution2D(4096, self.n_class, ksize= 4, stride=2, pad=0, nobias=True, 
                                      initialW=self.get_deconv_filter([4, 4, self.n_class, 4096])),
            upconv2=L.Deconvolution2D(self.n_class, self.n_class, ksize= 4, stride=2, pad=0, nobias=True,
                                      initialW=self.get_deconv_filter([4, 4, self.n_class, self.n_class])),
            upconv3=L.Deconvolution2D(self.n_class, self.n_class, ksize= 4, stride=2, pad=0, nobias=True,
                                      initialW=self.get_deconv_filter([4, 4, self.n_class, self.n_class])),
            upconv4=L.Deconvolution2D(self.n_class, self.n_class, ksize= 4, stride=2, pad=0, nobias=True,
                                      initialW=self.get_deconv_filter([4, 4, self.n_class, self.n_class])),
            upconv5=L.Deconvolution2D(self.n_class, self.n_class, ksize= 4, stride=2, pad=0, nobias=True,
                                      initialW=self.get_deconv_filter([4, 4, self.n_class, self.n_class])),            
        )
        self.train = True
        del self.wb

    @staticmethod
    def crop(inputs, outsize, offset):
        x = F.identity(inputs)
        crop_axis = [i!=j for i, j in zip(inputs.data.shape, outsize)]
        i = 0
        for index, tf in enumerate(crop_axis):
            if tf:
                _, x, _ = F.split_axis(x, [offset[i], offset[i] + outsize[index]], index)
                i += 1
        return x

    @staticmethod
    def calc_offset(in_shape, out_shape):
        return [(i - j) / 2 for i, j in zip(in_shape, out_shape) if i != j]

    @staticmethod
    def get_deconv_filter(f_shape):
        from math import ceil
        width = f_shape[0]
        heigh = f_shape[1]
        f = ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape, dtype=np.float32)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear
        return weights.transpose([3, 2, 0, 1])

    def __call__(self, x, t):
        h = self.predict_proba(x)
        self.loss = F.softmax_cross_entropy(h, t)
        #self.accuracy = self.calculate_accuracy(h, t)
        self.accuracy = F.accuracy(h, t, ignore_label=-1)
        self.IoU = self.calculate_intersection_of_union(h, t)
        return self.loss

    def predict(self, x):
        h = self.predict_proba(x)
        self.pred = F.softmax(h)
        return self.pred

    def predict_proba(self, x):
        output_shape = (x.data.shape[0], self.n_class, x.data.shape[2], x.data.shape[3])
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        p3 = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv4_1(p3))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        p4 = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv5_1(p4))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.dropout(F.relu(self.fc6_conv(h)), train=self.train, ratio=0.5)
        h = F.dropout(F.relu(self.fc7_conv(h)), train=self.train, ratio=0.5)

        h = F.relu(self.upconv1(h))
        p4 = self.upsample_pool4(p4)
        g = F.dropout(self.crop(p4, h.data.shape, self.calc_offset(p4.data.shape, h.data.shape)),
                      train=self.train, ratio=0.5)
        del p4
        h = F.relu(self.upconv2(h + g))
        p3 = self.upsample_pool3(p3)
        g = F.dropout(self.crop(p3, h.data.shape, self.calc_offset(p3.data.shape, h.data.shape)),
                      train=self.train, ratio=0.5)
        del p3
        j = F.relu(self.upconv3(h + g))

        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, stride=2)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        p2 = F.max_pooling_2d(h, 2, stride=2)
        p2 = self.upsample_pool2(p2)
        g = F.dropout(self.crop(p2, j.data.shape, self.calc_offset(p2.data.shape, j.data.shape)),
                      train=self.train, ratio=0.5)
        del p2
        j = F.relu(self.upconv4(j + g)) 

        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        p1 = F.max_pooling_2d(h, 2, stride=2)
        p1 = self.upsample_pool1(p1)
        g = F.dropout(self.crop(p1, j.data.shape, self.calc_offset(p1.data.shape, j.data.shape)),
                      train=self.train, ratio=0.5)
        del p1
        h = F.relu(self.upconv5(j + g))
        h = self.crop(h, output_shape, self.calc_offset(h.data.shape, output_shape))
        return h


    def calculate_accuracy(self, predictions, truths):
        if cuda.get_array_module(predictions.data) == cuda.cupy:
            with predictions.data.device:
                predictions =  predictions.data.get()
            with truths.data.device:
                truths = truths.data.get()
        else:
            predictions = predictions.data
            truths = truths.data

        # we want to exclude labels with -1
        mask = truths != -1
        # reduce values along classe axis
        reduced_preditions = np.argmax(predictions, axis=1) 
        # mask
        masked_reduced_preditions = reduced_preditions[mask]
        masked_truths = truths[mask] 
        s = (masked_reduced_preditions == masked_truths).mean()
        return s
        
    def calculate_intersection_of_union(self, predictions, truths):
        """ IoU metrics for human silhouette """
        predictions = predictions.data
        truths = truths.data
        xp = cuda.get_array_module(predictions)        
        mask1 = truths.reshape((truths.shape[0], truths.shape[1]*truths.shape[2])) > 0
        mask0 = predictions.argmax(axis=1).reshape(mask1.shape) > 0
        intersection = xp.logical_and(mask0, mask1).sum(axis=1) + 1
        union = xp.logical_or(mask0, mask1).sum(axis=1) + 1
        return (intersection.astype(predictions.dtype)  / union.astype(predictions.dtype)).mean()


def load_VGGmodel():
    print "loading VGG model..."
    if not os.path.exists(modelname):
        download()
    with open(modelname, 'rb') as d_pickle:
        data = six.moves.cPickle.load(d_pickle)
    return data


def download():
    print "Downloading pre-trained VGG16 model..."
    import wget
    wget.download(url)


'''

# caffe code
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512)
    n.pool5 = max_pool(n.relu5_3)

    # fully conv
    n.fc6, n.relu6 = conv_relu(n.pool5, 4096, ks=7, pad=0)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    n.fc7, n.relu7 = conv_relu(n.drop6, 4096, ks=1, pad=0)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
    n.score_fr = L.Convolution(n.drop7, num_output=21, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.upscore = L.Deconvolution(n.score_fr,
        convolution_param=dict(num_output=21, kernel_size=64, stride=32,
            bias_term=False),
        param=[dict(lr_mult=0)])
    n.score = crop(n.upscore, n.data)
    n.loss = L.SoftmaxWithLoss(n.score, n.label,
            loss_param=dict(normalize=False, ignore_label=255))

def copy_model(src, dst):
    assert isinstance(src, chainer.Chain)
    assert isinstance(dst, chainer.Chain)
    for child in src.children():
        if child.name not in dst.__dict__: continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child): continue
        if isinstance(child, chainer.Chain):
            copy_model(child, dst_child)
        if isinstance(child, chainer.Link):
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    match = False
                    break
            if not match:
                print 'Ignore %s because of parameter mismatch' % child.name
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data
            print 'Copy %s' % child.name


'''
