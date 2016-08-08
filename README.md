# Deep Learning for Human Part Discovery in Images - Chainer implementation

NOTE: This is not official implementation. Original paper is [Deep Learning for Human Part Discovery in Images](http://lmb.informatik.uni-freiburg.de/Publications/2016/OB16a/oliveira16icra.pdf).

# Requirements

- Python 2.7.11+

  - [Chainer 1.10+](https://github.com/pfnet/chainer) (Neural network framework)
  - numpy 1.9+
  - scipy 0.16+
  - tqdm
  - six
  - scipy 0.16+


# Data preparation

```
bash prepare.sh
```

This script downloads VOC 2010 dataset (<>) and the authors' original dataset (<>).


# Start training

Starting with the prepared shells is the easiest way. If you want to run `train.py` with your own settings, please check the options first by `python scripts/train.py --help` and modify one of the following shells to customize training settings.


## For MPII Dataset

```
bash shells/train_mpii.sh
```

### GPU memory requirement

- AlexNet

  - batchsize: 128 -> about 2870 MiB
  - batchsize: 64 -> about 1890 MiB
  - batchsize: 32 (default) -> 1374 MiB


# Visualize Prediction

## Example

### Prediction and visualize them and calc mean errors

```
```

### Tile some randomly selected result images

```
```

### Create animated GIF to intuitively compare predictions and labels

```
```








