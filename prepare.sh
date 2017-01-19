# download
mkdir data/img
mkdir data/mask

wget http://www2.informatik.uni-freiburg.de/~oliveira/datasets/Sitting.tar.gz
wget http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/trainval.tar.gz
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
# wget https://drive.google.com/open?id=0BxSyYt1jT6LhUlhITjdicDFyNHM -O fcn-8s-pascalcontext_W_and_b.pkl
# wget doesn't work on google drive address. please download via your blowser.

# untar
tar zxvf Sitting.tar.gz
tar zxvf trainval.tar.gz
tar xvf VOCtrainval_03-May-2010.tar

mv img/* data/img/
# mv VOCdevkit/VOC2010/JPEGImages/* data/img/   # this doesn't work because arguments too long. instead:
find VOCdevkit/VOC2010/JPEGImages/ -name '*.jpg' -print0 | xargs -0 -I{} mv {} data/img
mv masks/* data/mask/
# mv Annotations_Part/* data/mask/      # this doesn't work because arguments too long. instead:
find Annotations_Part/ -name '*.mat' -print0 | xargs -0 -I{} mv {} data/mask/
# mv VOCdevkit/ data

rm *.m
rm Sitting.tar.gz
rm trainval.tar.gz
rm VOCtrainval_03-May-2010.tar
rmdir Annotations_Part
rmdir img
rmdir masks
