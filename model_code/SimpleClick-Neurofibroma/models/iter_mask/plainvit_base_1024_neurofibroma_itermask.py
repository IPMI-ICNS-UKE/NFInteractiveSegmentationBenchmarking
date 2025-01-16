import os
from isegm.utils.exp_imports.default import *
from isegm.model.modeling.transformer_helper.cross_entropy_loss import CrossEntropyLoss
from isegm.data.datasets.neurofibroma import NeurofibromaDataset


# Constants and configurations specific to fine-tuning
# Neurofibroma dataset.
IMG_SIZE = (1024, 1024)
EPOCH_LEN = 10000
VAL_EPOCH_LEN = 2000
MIN_OBJECT_AREA = 100
NUM_EPOCHS = 55
IMAGE_DUMP_INTERVAL = 300
CHECKPOINT_INTERVAL = [(0, 20), (50, 1)]
MILESTONES = [50, 55]
DEFAULT_LR = 5e-5
DEFAULT_BETAS = (0.9, 0.999)
DEFAULT_EPS = 1e-8


def main(cfg):
    """Main function to initialize the model and start training."""
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    """Initialize the PlainVitModel with the provided configuration."""
    model_cfg = edict()
    model_cfg.crop_size = IMG_SIZE
    model_cfg.num_max_points = 24
    
    backbone_params = dict(
        img_size=model_cfg.crop_size,
        patch_size=(16,16),
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4, 
        qkv_bias=True,
    )

    neck_params = dict(
        in_dim = 768,
        out_dims = [128, 256, 512, 1024],
    )

    head_params = dict(
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        dropout_ratio=0.1,
        num_classes=1,
        loss_decode=CrossEntropyLoss(),
        align_corners=False,
        upsample=cfg.upsample,
        channels={'x1':256, 'x2': 128, 'x4': 64}[cfg.upsample],
    )

    model = PlainVitModel(
        use_disks=True,
        norm_radius=5,
        with_prev_mask=True,
        backbone_params=backbone_params,
        neck_params=neck_params,
        head_params=head_params,
        random_split=cfg.random_split,
    )
    
    assert cfg.MAE_BASE, "Pretrained weights path is required."
    model.backbone.init_weights_from_pretrained(cfg.MAE_BASE)
    model.to(cfg.device)

    return model, model_cfg


def train(model, cfg, model_cfg):
    """Train the model with the provided datasets and configurations."""
    cfg.batch_size = 32 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0
    
    # Data augmentation
    train_augmentator = Compose([
        HorizontalFlip(),
        ShiftScaleRotate(shift_limit=0.02, scale_limit=0.1, rotate_limit=10),
        Resize(*IMG_SIZE)
    ], p=1.0)

    val_augmentator = Compose([
        Resize(*IMG_SIZE)
    ], p=1.0)

    points_sampler = MultiPointSampler(
        model_cfg.num_max_points, 
        prob_gamma=0.80,
        merge_objects_prob=0.15,
        max_num_merged_objects=2
    )
    
    # Datasets initialization
    trainset = NeurofibromaDataset(
        cfg.DATASET_PATH,
        split=os.path.join(cfg.FOLD_PATH, f"fold_{cfg.fold}", "train_set.txt"),
        augmentator=train_augmentator,
        min_object_area=MIN_OBJECT_AREA,
        keep_background_prob=0.05,
        points_sampler=points_sampler,
        epoch_len=EPOCH_LEN
    )
    
    valset = NeurofibromaDataset(
        cfg.DATASET_PATH,
        split=os.path.join(cfg.FOLD_PATH, f"fold_{cfg.fold}", "val_set.txt"),
        augmentator=val_augmentator,
        min_object_area=MIN_OBJECT_AREA,
        points_sampler=points_sampler,
        epoch_len=VAL_EPOCH_LEN
    )
    
    optimizer_params = {
        'lr': DEFAULT_LR,
        'betas': DEFAULT_BETAS,
        'eps': DEFAULT_EPS,
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=MILESTONES, gamma=0.1)
    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        layerwise_decay=cfg.layerwise_decay,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=CHECKPOINT_INTERVAL,
                        image_dump_interval=IMAGE_DUMP_INTERVAL,
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=3)
    trainer.run(num_epochs=NUM_EPOCHS, validation=False)
