import argparse
import torch
import os, sys
import numpy as np

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

sys.path.append(os.path.abspath(".."))
# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
# import trainers.cocoop
# import trainers.zsclip
# import trainers.maple
# import trainers.independentVL
# import trainers.vpt
import eft.trainers.rada
import trainers.clip_adapter
import trainers.query_key_coop
from clip import clip, tokenize, load


import wandb

DOWNLOAD_ROOT_v2 = '/l/users/zhiqiang.shen/ghazi/DATA/CLIP'


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME

    model, _, _ = load(backbone_name, device='cpu', download_root=DOWNLOAD_ROOT_v2)

    return model


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head
    
    if args.layers:
        cfg.TRAINER.QKMASK.PROJ_LAYERS=args.layers

    if args.lr:
        cfg.OPTIM.LR = args.lr

    if args.bs:
        cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.bs

    if args.ep:
        cfg.OPTIM.MAX_EPOCH = args.ep

    if args.alpha:
        cfg.ALPHA = args.alpha

    if args.lr1:
        cfg.LR1 = args.lr1

    if args.temp:
        cfg.TEMP = args.temp

    if args.gpu:
        cfg.GPU = args.gpu


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    # optimizer
    cfg.ALPHA = 0.0
    cfg.TEMP = 1.0
    cfg.LR1 = 1e-2
    cfg.GPU = 0

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for QKMASK
    cfg.TRAINER.QKMASK = CN()
    cfg.TRAINER.QKMASK.N_CTX = 2  # number of context vectors
    cfg.TRAINER.QKMASK.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.QKMASK.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.QKMASK.PROJ_LAYERS = 1
    cfg.TRAINER.QKMASK.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for independent Vision Language prompting (independent-vlp)
    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.IVLP.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 9 # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for only vision side prompting
    cfg.TRAINER.VPT = CN()
    cfg.TRAINER.VPT.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.VPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.VPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.VPT.PROMPT_DEPTH_VISION = 1  # if set to 1, will represent shallow vision prompting only
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new


    cfg.TRAINER.MASKCOOP = CN()
    cfg.TRAINER.MASKCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.MASKCOOP.CSC = False  # class-specific context
    cfg.TRAINER.MASKCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.MASKCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.MASKCOOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MASKCOOP.PROJ_LAYERS = 1
    cfg.TRAINER.MASKCOOP.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for LP
    cfg.TRAINER.CLIP_ADAPTER = CN()
    cfg.TRAINER.CLIP_ADAPTER.N_CTX = 2  # number of context vectors
    cfg.TRAINER.CLIP_ADAPTER.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.CLIP_ADAPTER.PREC = "fp16"  # fp16, fp32, amp
    # cfg.TRAINER.QKMASK.PROJ_LAYERS = 1
    # cfg.TRAINER.QKMASK.PROMPT_DEPTH = 9 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "base"  # all, base or new
    cfg.DATASET.NUM_SHOTS = 16


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):

    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    #  # Initialize W&B project
    # wandb.init(project="stanford_cars_lr5e-3_ep=8_bs10_alpha=0_layers=1")


    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    if args.split == "test":
        cfg.defrost()

        # Now, modify the attribute
        cfg.DATASET.SUBSAMPLE_CLASSES = "new"

        # Lock the config node again to prevent further modification
        cfg.freeze()
        
    trainer = build_trainer(cfg)

    if args.split == "train":
        data_loader = trainer.train_loader_x
    elif args.split == "val":
        data_loader = trainer.test_loader
    elif args.split == "test":
        data_loader = trainer.test_loader
        
    print(f"DataLoader: {len(data_loader.dataset)}")
    ########################################
    #   Setup Network
    ########################################
    clip_model = load_clip_to_cpu(cfg)
    clip_model = clip_model.cuda()
    clip_model.eval()
    ###################################################################################################################
    # Start Feature Extractor
    feature_list = []
    label_list = []
    train_dataiter = iter(data_loader)
    for train_step in range(1, len(train_dataiter) + 1):
        batch = next(train_dataiter)
        data = batch["img"].cuda()
        feature = clip_model.visual(data)
        feature = feature.cpu()
        for idx in range(len(data)):
            feature_list.append(feature[idx].tolist())
        label_list.extend(batch["label"].tolist())
    save_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.NAME)
    os.makedirs(save_dir, exist_ok=True)
    save_filename = f"{args.split}"
    np.savez(
        os.path.join(save_dir, save_filename),
        feature_list=feature_list,
        label_list=label_list,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], help="which split")

    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    parser.add_argument(
        "--layers", type=int, default=1, help="number of projection layers"
    )
    parser.add_argument(
        "--bs", type=int, default=100, help="training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="learning rate"
    )
    parser.add_argument(
        "--ep", type=int, default=10, help="total epochs to train"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.0, help="consatnt factor for the mask loss"
    )
    parser.add_argument(
        "--lr1", type=float, default=1e-2, help="lr for mask weights"
    )
    parser.add_argument(
        "--temp", type=float, default=1.0, help="temperature for self-attention"
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU used for training"
    )
    args = parser.parse_args()
    main(args)




