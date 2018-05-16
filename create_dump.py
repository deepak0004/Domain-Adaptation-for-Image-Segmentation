import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab_multi import Res_Deeplab
from model.deeplab_multi import getVGG
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.gta5_dataset import GTA5DataSet
from collections import OrderedDict
import os
from PIL import Image

import matplotlib.pyplot as plt
import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = './GTA5'#'./Cityscape'#
DATA_LIST_PATH = "./dataset/gta5_list/train.txt"#"./dataset/cityscapes_list/train.txt"#
SAVE_PATH = './result/cityscapes_vgg'

IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500 # Number of images in the validation set.
RESTORE_FROM = "./weights/GTA2Cityscapes_multi-ed35151c.pth"
#RESTORE_FROM = "./snapshots_vgg/GTA5_VGG_35000.pth"
SET = 'train'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()


def main():
    
    
    city = np.load("dump_cityscape5.npy")
    gta = np.load("dump_gta5.npy")

    city_scape = city[:1000,:]
    gta5 = gta[:1000,:]

    combined = np.concatenate((city[1000:,:], gta[1000:,:]))

    np.save('source.npy', gta5)
    np.save('target.npy', city_scape)
    np.save('mixed.npy', combined)
    print(city_scape.shape)
    print(gta5.shape)
    print(combined.shape)
    exit()

    print(type(dump))
    print(dump.shape)

    
    b = dump
    print(type(dump))
    print(dump.shape)

    import random

    a = np.stack(random.sample(a, 500))
    b = np.stack(random.sample(b, 500))
    
    dump = np.concatenate((a,b))
    print(dump.shape)

    arr = np.arange(10)
    #print(dump)
    exit()
    
    





    """Create the model and start the evaluation process."""

    args = get_arguments()

    gpu0 = args.gpu

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    model = Res_Deeplab(num_classes=args.num_classes)
    #model = getVGG(num_classes=args.num_classes)

    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda(gpu0)

    #trainloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set), batch_size=1, shuffle=False, pin_memory=True)

    trainloader = data.DataLoader(GTA5DataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False), batch_size=1, shuffle=False, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    interp = nn.Upsample(size=(1024, 2048), mode='bilinear')
    dump_array = np.array((1,2))

    for itr in xrange(2000):
        print(itr)
        _, batch = trainloader_iter.next()
        images, labels, _, _ = batch
        #images, _, _ = batch

        output1, output2 = model(Variable(images, volatile=True).cuda(gpu0))
        import torch.nn.functional as F
        output2 = F.avg_pool2d(output2, (4, 4))
        output2 = output2.data.cpu().numpy()
        output2 = np.reshape(output2, (1, -1))

        if dump_array.shape == (2,):
            dump_array = output2
        else:
            dump_array = np.concatenate((dump_array, output2))

    np.save('dump_gta5.npy', dump_array)


if __name__ == '__main__':
    main()
