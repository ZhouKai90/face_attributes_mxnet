# -*- coding: UTF-8 -*-

from easydict import EasyDict as edict

fa_config = edict()

fa_config.base_lr = 0.1
fa_config.gpus = '0, 1'
fa_config.num_epoch = 300
fa_config.batch_size = 128
fa_config.num_examples = 51246

fa_config.img_channles = 3
fa_config.img_height = 112
fa_config.img_width = 112
fa_config.num_class = 5


fa_config.bn_mom = 0.9
fa_config.workspace = 256
fa_config.act_type = 'prelu'


fa_config.root_path = '/kyle/workspace/project/face_attributes_prediction'
fa_config.log_path = fa_config.root_path + '/log/train.log'

#net for train
fa_config.model_path = fa_config.root_path + '/model/resmobile'

#for load dataset from .rec file
fa_config.dataset_path =  fa_config.root_path + '/data/'
fa_config.imgrec_train = fa_config.dataset_path + 'single_anno_train.rec'
fa_config.imgrec_val = fa_config.dataset_path + 'single_anno_val.rec'
fa_config.imgidx_train = fa_config.dataset_path + 'single_anno_train.idx'
fa_config.imgidx_val = fa_config.dataset_path + 'single_anno_val.idx'
