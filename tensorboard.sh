#!/bin/bash

export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
sudo dpkg-reconfigure locales

source activate my-env

tensorboard  --logdir=log/train
