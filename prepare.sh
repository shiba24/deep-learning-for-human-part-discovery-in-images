# download
mkdir data/img
mkdir data/mask

wget http://www2.informatik.uni-freiburg.de/~oliveira/datasets/Sitting.tar.gz
wget http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/trainval.tar.gz
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar

# untar
tar zxvf Sitting.tar.gz
tar zxvf trainval.tar.gz
tar xvf VOCtrainval_03-May-2010.tar

mv img/* data/img/
mv VOCdevkit/VOC2010/JPEGImages/* data/img/
mv masks/* data/mask/
mv Annotations_Part/* data/mask/

rm *.m
rmdir Annotations_Part
rmdir img
rmdir masks
