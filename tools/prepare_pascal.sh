#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REC_DATA_PATH=$DIR/../data/face_attributes_rec

python $DIR/prepare_dataset.py \
    --dataset pascal \
    --set train \
    --target $REC_DATA_PATH/train.lst \
    --root $DIR/../data/devkit \


python $DIR/prepare_dataset.py \
    --dataset pascal \
    --set val \
    --target $REC_DATA_PATH/val.lst \
    --root $DIR/../data/devkit \
