from __future__ import print_function
from __future__ import unicode_literals
from .collect import Config
import copy


cfg = Config()

cfg.exp_name = 'test_ucf_demo'

cfg.data.hmdb.train_len = 847 # 11988 # 6090
cfg.data.hmdb.val_len = 94 # 1332 # 676

cfg.dataset.hmdb.ann_file = '/home/aistudio/TPN/data/ucf_demo/train.txt' # '/home/aistudio/TPN/data/HMDB_51/train.txt'
cfg.dataset.hmdb.img_prefix = '/home/aistudio/TPN/data'
cfg.dataset.hmdb.img_norm_cfg.mean = [123.675, 116.28, 103.53]
cfg.dataset.hmdb.img_norm_cfg.std = [58.395, 57.12, 57.375]
cfg.dataset.hmdb.img_norm_cfg.to_rgb = True
cfg.dataset.hmdb.num_segments = 1
cfg.dataset.hmdb.new_length = 32
cfg.dataset.hmdb.new_step = 2
cfg.dataset.hmdb.random_shift = True
cfg.dataset.hmdb.modality = 'RGB'
cfg.dataset.hmdb.image_tmpl = 'img_{}.jpg'
cfg.dataset.hmdb.img_scale = 256
cfg.dataset.hmdb.resize_keep_ratio = True
cfg.dataset.hmdb.input_size = 224
cfg.dataset.hmdb.flip_ratio = 0.5
cfg.dataset.hmdb.oversample = None
cfg.dataset.hmdb.resize_crop = True
cfg.dataset.hmdb.color_jitter = True
cfg.dataset.hmdb.color_space_aug = True
cfg.dataset.hmdb.max_distort = 0
cfg.dataset.hmdb.test_mode = False
cfg.dataset.hmdb.input_format = 'NCTHW'
cfg.dataset.hmdb.num_threads = 2
cfg.dataset.hmdb.buf_size = 1024

cfg.models.backbone_name = 'resnet'

cfg.models.backbone.depth = 50
cfg.models.backbone.num_stages = 4
cfg.models.backbone.out_indices = [2, 3]
cfg.models.backbone.inflate_freqs = (0, 0, 1, 1)

cfg.models.tpn.in_channels = [1024, 2048]
cfg.models.tpn.out_channels = 1024

cfg.models.tpn.spatial_modulation_config.inplanes = [1024, 2048]
cfg.models.tpn.spatial_modulation_config.planes = 2048

cfg.models.tpn.temporal_modulation_config.scales = (32, 32)
cfg.models.tpn.temporal_modulation_config.param.inplanes = -1
cfg.models.tpn.temporal_modulation_config.param.planes = -1
cfg.models.tpn.temporal_modulation_config.param.downsample_scale = -1

cfg.models.tpn.upsampling_config.scales = (1, 1, 1)

cfg.models.tpn.downsampling_config.scales = (1, 1, 1)
cfg.models.tpn.downsampling_config.param.inplanes = -1
cfg.models.tpn.downsampling_config.param.planes = -1
cfg.models.tpn.downsampling_config.param.downsample_scale = -1

cfg.models.tpn.level_fusion_config.in_channels = [1024, 1024]
cfg.models.tpn.level_fusion_config.mid_channels = [1024, 1024]
cfg.models.tpn.level_fusion_config.out_channels = 2048
cfg.models.tpn.level_fusion_config.ds_scales=[(1, 1, 1), (1, 1, 1)]

cfg.models.tpn.aux_head_config.inplanes = -1
cfg.models.tpn.aux_head_config.planes = 101
cfg.models.tpn.aux_head_config.loss_weight = 0.5


cfg.models.cls_head.with_avg_pool = False
cfg.models.cls_head.temporal_feature_size = 1
cfg.models.cls_head.spatial_feature_size = 1
cfg.models.cls_head.dropout_ratio = 0.5
cfg.models.cls_head.in_channels = 2048
cfg.models.cls_head.num_classes = 101 

cfg.solver.log_dir = '/home/aistudio/TPN/logs'
cfg.solver.save_dir = '/home/aistudio/TPN/saved_models'
cfg.solver.start_epoch = 0
cfg.solver.max_epoch = 151
cfg.solver.batch_size = 8
cfg.solver.lr_decay_epoch = [75, 125]
cfg.solver.base_lr = 0.01
cfg.solver.momentum_rate = 0.9
cfg.solver.l2_decay = 1e-4
cfg.solver.log_interval = 4
cfg.solver.save_interval = 10
cfg.solver.eval_interval = 5

cfg.eval.solver.batch_size = 1

cfg.eval.dataset.hmdb.ann_file = '/home/aistudio/TPN/data/ucf_demo/val.txt' # '/home/aistudio/TPN/data/HMDB_51/val.txt'
cfg.eval.dataset.hmdb.img_prefix = '/home/aistudio/TPN/data'
cfg.eval.dataset.hmdb.img_norm_cfg.mean = [123.675, 116.28, 103.53]
cfg.eval.dataset.hmdb.img_norm_cfg.std = [58.395, 57.12, 57.375]
cfg.eval.dataset.hmdb.img_norm_cfg.to_rgb = True
cfg.eval.dataset.hmdb.num_segments = 5
cfg.eval.dataset.hmdb.new_length = 32
cfg.eval.dataset.hmdb.new_step = 2
cfg.eval.dataset.hmdb.random_shift = True
cfg.eval.dataset.hmdb.modality = 'RGB'
cfg.eval.dataset.hmdb.image_tmpl = 'img_{}.jpg'
cfg.eval.dataset.hmdb.img_scale = 256
cfg.eval.dataset.hmdb.resize_keep_ratio = True
cfg.eval.dataset.hmdb.input_size = 224
cfg.eval.dataset.hmdb.flip_ratio = 0
cfg.eval.dataset.hmdb.resize_crop = True
cfg.eval.dataset.hmdb.color_jitter = False
cfg.eval.dataset.hmdb.color_space_aug = False
cfg.eval.dataset.hmdb.oversample = None
cfg.eval.dataset.hmdb.max_distort = 0
cfg.eval.dataset.hmdb.test_mode = False
cfg.eval.dataset.hmdb.input_format = 'NCTHW'
cfg.eval.dataset.hmdb.num_threads = 2
cfg.eval.dataset.hmdb.buf_size = 1024
