import torch
import argparse
import shutil
import os, sys
from pathlib import Path

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part import train

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix

if os.getcwd() + '/MRAugment/mraugment/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/MRAugment/mraugment/')
from MRAugment.mraugment.data_augment import DataAugmentor

def parse():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/Data/val/', help='Directory of validation data')
    
    parser.add_argument('--cascade', type=int, default=1, help='Number of cascades | Should be less than 12') ## important hyperparameter
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net | 18 in original varnet') ## important hyperparameter
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net | 8 in original varnet') ## important hyperparameter
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')
    
    #0711 추가
    parser = DataAugmentor.add_augmentation_specific_args(parser)

    args = parser.parse_args()
    args.max_epochs = args.num_epochs
    return args

if __name__ == '__main__':
    args = parse()
    
    # Print all arguments
    print("Training Parameters:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    
    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    args.exp_dir = Path('../result') / args.net_name / 'checkpoints'
    args.val_dir = Path('../result') / args.net_name / 'reconstructions_val'
    args.main_dir = Path('../result') / args.net_name / __file__
    args.val_loss_dir = Path('../result') / args.net_name

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)
    
    # data augmentaion part 추가    
    current_epoch = [0]  # Mutable object to store current epoch
    
    def current_epoch_fn(): # lambda 에러 수정 위해 추가
        return current_epoch[0]
    args.current_epoch_fn = current_epoch_fn
    augmentor = DataAugmentor(args, current_epoch_fn, args.seed) # 0713 seed fix추가

    train(args, augmentor)
