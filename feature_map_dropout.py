import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class FeatureMapDropout(function.Function):

    """Feature Map Dropout regularization."""

    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].ndim >= 2) # need spatial dim beside batch x features x ...
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        if not hasattr(self, 'mask'):
            scale = x[0].dtype.type(1. / (1 - self.dropout_ratio))
            xp = cuda.get_array_module(*x)
            # interpret first two axes as feature channel
            if xp == numpy:
                flag = xp.random.rand(*x[0].shape[:2]) >= self.dropout_ratio
            else:
                flag = (xp.random.rand(*x[0].shape[:2], dtype=numpy.float32) >=
                        self.dropout_ratio)
            self.mask = scale * flag
        #TODO: check, if broadcast works as intended
        return x[0] * self.mask,

    def backward(self, x, gy):
        return gy[0] * self.mask,

def feature_map_dropout(x, ratio=.5, train=True):
    """Drops feature maps of input variable randomly.

    This function drops whole spatial feature maps randomly with probability ``ratio`` and
    scales the remaining elements by factor ``1 / (1 - ratio)``. In testing
    mode, it does nothing and just returns ``x``.

    Args:
        x (~chainer.Variable): Input variable.
        ratio (float): Dropout ratio.
        train (bool): If ``True``, executes dropout. Otherwise, does nothing.

    Returns:
        ~chainer.Variable: Output variable.

    See the paper by J. Tompson: `Efficient object localization using convolutional networks \
    <https://arxiv.org/abs/1411.4280>`_.

    """
    if train:
        return FeatureMapDropout(ratio)(x)
    return x