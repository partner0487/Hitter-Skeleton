
default_scope = 'mmpose'

# custom hooks
custom_hooks = [
    # Synchronize model buffers such as running_mean and running_var in BN
    # at the end of each epoch
    dict(type='SyncBuffersHook')
]

# multi-processing backend
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)


# logger
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
log_level = 'INFO'
load_from = None
resume = False

# file I/O backend
backend_args = dict(backend='local')



# base dataset settings
data_root = "data/"
dataset_type = "Halpe26Dataset"
data_mode = "topdown"
num_keypoints = 26

# codec settings
codec = dict(type="UDPHeatmap", input_size=(192, 256), heatmap_size=(48, 64), sigma=2)
# runtime
train_cfg = dict(max_epochs=100, val_interval=4)


# hooks
default_hooks = dict(
    # checkpoint=dict(interval=5, save_best="coco/AP", rule="greater", max_keep_ckpts=1),
    checkpoint=dict(interval=5, save_best="coco/AP", rule="greater", max_keep_ckpts=1),
    logger=dict(type="LoggerHook", interval=100),
)

# optimizer
custom_imports = dict(
    imports=["mmpose.engine.optim_wrappers.layer_decay_optim_wrapper"],
    allow_failed_imports=False,
)

optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.75,
        custom_keys={
            "bias": dict(decay_multi=0.0),
            "pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        },
    ),
    constructor="LayerDecayOptimWrapperConstructor",
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

# learning policy
param_scheduler = [
    dict(
        type="LinearLR", begin=0, end=500, start_factor=0.001, by_epoch=False
    ),  # warm-up
    dict(
        type="MultiStepLR",
        begin=0,
        end=100,
        milestones=[60, 90],
        gamma=0.1,
        by_epoch=True,
    ),
]

# automatically scaling LR based on the actual training batch size
# auto_scale_lr = dict(base_batch_size=512)


# model settings
model = dict(
    type="TopdownPoseEstimator",
    backbone=dict(
        type="mmpretrain.VisionTransformer",
        arch="base",
        img_size=(256, 192),
        patch_size=16,
        qkv_bias=True,
        drop_path_rate=0.3,
        out_type="featmap",
        patch_cfg=dict(padding=2),
        frozen_stages=9,
        init_cfg=dict(
            type="Pretrained",
            checkpoint=rf"checkpoint\td-hm_ViTPose-base-simple_20240201.pth",
        ),
        with_cls_token=False,
    ),
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    neck=dict(type="FeatureMapProcessor", scale_factor=4.0, apply_relu=True),
    head=dict(
        decoder=dict(
            heatmap_size=(
                48,
                64,
            ),
            input_size=(
                192,
                256,
            ),
            sigma=2,
            type="UDPHeatmap",
        ),
        deconv_kernel_sizes=[],
        deconv_out_channels=[],
        final_layer=dict(kernel_size=3, padding=1),
        in_channels=768,
        loss=dict(type="KeypointMSELoss", use_target_weight=True),
        out_channels=num_keypoints,
        type="HeatmapHead",
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode="heatmap",
        shift_heatmap=False,
    ),
)


# pipelines
train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomHalfBody"),
    dict(type="RandomBBoxTransform"),
    dict(type="TopdownAffine", input_size=codec["input_size"], use_udp=True),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]
val_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=codec["input_size"], use_udp=True),
    dict(type="PackPoseInputs"),
]

halpe_halpe26 = [(i, i) for i in range(num_keypoints)]

# key_convert_pipe = [dict(
#     type="KeypointConverter", num_keypoints=num_keypoints, mapping=halpe_halpe26
# )]


# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        metainfo=dict(from_file="configs/_base_/datasets/halpe26.py"),
        ann_file="halpe/annotations/halpe_train_v1_26.json",
        data_prefix=dict(img="halpe/images/train2015"),
        pipeline=train_pipeline,
    ),
)


# val_dataloader = dict(
#     batch_size=32,
#     num_workers=4,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_mode=data_mode,
#         metainfo=dict(from_file="configs/_base_/datasets/halpe26.py"),
#         ann_file="halpe/annotations/halpe_val_v1_26.json",
#         bbox_file=None,
#         data_prefix=dict(img="coco/val2017/"),
#         pipeline=val_pipeline,
#         test_mode=True,
#     ),
# )
test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        metainfo=dict(from_file="configs/_base_/datasets/halpe26.py"),
        ann_file="halpe/annotations/halpe_val_v1_26.json",
        bbox_file=None,
        data_prefix=dict(img="coco/val2017/"),
        pipeline=val_pipeline,
        test_mode=True,
    ),
)




# evaluators
# test_evaluator = [dict(type="CocoMetric")]
# val_evaluator = dict(
#     type="CocoMetric", ann_file=data_root + "halpe/annotations/halpe_val_v1.json"
# )
val_cfg=None
test_evaluator =  dict(type="PCKAccuracy")
# test_evaluator = [
#     dict(type="PCKAccuracy"),
#     # dict(
#     #     type="CocoMetric", ann_file=data_root + "halpe/annotations/halpe_val_v1_26.json"
#     # ),
# ]

# test_evaluator = val_evaluator

visualizer = dict(
    vis_backends=[
        dict(type="LocalVisBackend"),
        dict(type="TensorboardVisBackend"),
    ]
)
