#!/bin/bash

source activate my-env

export LC_ALL=C

tensorboard  --logdir=log_gan
