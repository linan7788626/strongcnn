#!/bin/bash

# change to the directory
cd /home/cpd/Projects/strongcnn/catalog/

# download catalogs
wget http://slac.stanford.edu/cpd/galzoo.tar
wget http://slac.stanford.edu/cpd/swap_catalog.tar

tar xvf galzoo.tar
unzip 'galzoo/*.zip'
rm galzoo/*.zip
tar xvf swap_catalog.tar

# set up data
mkdir caffe_data

# load up the caffe code and such
myml

# galzoo
python /home/cpd/Projects/strongcnn/code/end_to_end/prep_galzoo.py
convert_imageset.bin caffe_data/galzoo_processed caffe_data/galzoo_train_listfile caffe_data/galzoo_train_lmdb
convert_imageset.bin caffe_data/galzoo_processed caffe_data/galzoo_val_listfile caffe_data/galzoo_val_lmdb
compute_image_mean.bin caffe_data/galzoo_train_lmdb caffe_data/galzoo_train_mean_image -backend "lmdb"

# spacewarps
# NOTE that prep_swap does not process images. just takes them as is!
python /home/cpd/Projects/strongcnn/code/end_to_end/prep_swap.py
convert_imageset.bin cutout_catalog/cutouts caffe_data/sw_train_listfile caffe_data/sw_train_lmdb
convert_imageset.bin cutout_catalog/cutouts caffe_data/sw_val_listfile caffe_data/sw_val_lmdb
compute_image_mean.bin caffe_data/sw_train_lmdb caffe_data/sw_train_mean_image -backend "lmdb"



