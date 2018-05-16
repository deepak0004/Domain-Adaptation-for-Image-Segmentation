import numpy as np
import argparse
import json
from PIL import Image
from os.path import join


dictt = {(0,  0,  0): 255, (111, 74, 0): 255, ( 81,  0, 81): 255, (128, 64,128): 0, 
(244, 35,232): 1, (250,170,160): 255, (230,150,140): 255, (70, 70, 70): 2, (102,102,156): 3, 
(190,153,153): 4,  (180,165,180): 255, (150,100,100): 255, (150,120, 90):255, (153,153,153): 5, 
(153,153,153): 255, (250,170,30): 6, (220,220,0): 7, (107,142, 35): 8, (152,251,152): 9, 
(70,130,180): 10, (220, 20, 60): 11, (255, 0, 0): 12, (0, 0, 142): 13, (0, 0, 70): 14, (0, 60, 100): 15, 
(0, 0, 90) : 255, (0, 0, 110):255, (0, 80, 100): 16, (0, 0, 230):17, (119, 11, 32):18,  
(0, 0, 142): 19}

#dictt2 = { (0, 128, 64): 255, (128, 0, 0): 2, (64, 0, 128): 13, (64, 64, 128): 4, 
#(192, 0, 192): 17, (64, 192, 128): 11, (128, 64, 128): 0, (0, 0, 192): 1, (128, 128, 128): 10, 
#(64, 128, 192): 14, (192, 64, 128): 16, (64, 0, 64): 255, (192, 192, 0): 8, (64, 192, 0): 3}

dictt2 = {(128, 64, 128): 0, (0, 0, 192): 1, (128, 0, 0): 2, (64, 192, 0): 3, 
(64, 64, 128): 4, (192, 192, 128): 5, (0, 64, 64): 6, (128, 128, 0): 7, 
(192, 192, 0): 7, (128, 128, 128): 8, (64, 64, 0): 9, (0, 128, 192): 10, 
(64,128,192): 11, (64, 128, 192): 12, (192,128,192): 13, (192,0,192): 14, 
(128,0,192): 15, (192, 0, 64): 16}

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def label_mapping2(input, mapping):
    output = np.copy(input)
    oioi = []
    for i in range(1024):
        for j in range(2048):
            tup = (output[i][j][0], output[i][j][1], output[i][j][2])
            oioi.append( dictt[tup] )
    #for ind in range(len(mapping)):
    #    output[input == dictt[ind][0]] = dictt[ind][1]
    return np.array(oioi, dtype=np.int64)

def label_mapping2(input, mapping):
    output = np.copy(input)
    oioi = []
    for i in range(1024):
        for j in range(2048):
            tup = (output[i][j][0], output[i][j][1], output[i][j][2])
            if( tup in dictt ):
               oioi.append( dictt[tup] )
            else:
               oioi.append( 255 ) 
    #for ind in range(len(mapping)):
    #    output[input == dictt[ind][0]] = dictt[ind][1]
    return np.array(oioi, dtype=np.int64)

def label_mapping3(input, mapping):
    output = np.copy(input)
    oioi = []
    for i in range(1024):
        for j in range(2048):
            tup = (output[i][j][0], output[i][j][1], output[i][j][2])
            if( tup in dictt2 ):
               oioi.append( dictt2[tup] )
            else:
               oioi.append( 255 ) 
    #for ind in range(len(mapping)):
    #    output[input == dictt[ind][0]] = dictt[ind][1]
    return np.array(oioi, dtype=np.int64)

def compute_mIoU(gt_dir, pred_dir, devkit_dir=''):
    """
    Compute IoU given the predicted colorized images and 
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
      info = json.load(fp)

    num_classes = np.int(info['classes'])
    print('Num classes', num_classes)
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int)
    hist = np.zeros((num_classes, num_classes))

    #image_path_list = 'Orig/'
    #label_path_list = join(devkit_dir, 'label.txt')
    orig_val = 'ground_iou.txt'
    pred_val = 'pred_iou.txt'
    gt_imgs = open(orig_val, 'r').read().splitlines()
    gt_imgs = [join(gt_dir, x.split('/')[-1]) for x in gt_imgs]
    pred_imgs = open(pred_val, 'r').read().splitlines()
    pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]

    for ind in range(len(gt_imgs)):
        print 'Ind: ', gt_imgs[ind]
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))
        print label.shape
        #print 'label: ', label
        #print 'pred: ', pred
        '''
        for i in range(1000):
            for j in range(1000):
               if( label[i][j][0]!=0 or label[i][j][1]!=0 or label[i][j][2]!=0 ):
                  print 'yo'
        '''
        #label = label_mapping(pred, mapping)
        label = label_mapping2(label, mapping)
        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))
    
    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    return mIoUs


#def main(args):
compute_mIoU('Ground', 'Pred', 'dataset/cityscapes_list')

'''
parser = argparse.ArgumentParser()
parser.add_argument('Ground', type=str, help='directory which stores CityScapes val gt images')
parser.add_argument('Pred', type=str, help='directory which stores CityScapes val pred images')
parser.add_argument('--devkit_dir', default='dataset/cityscapes_list', help='base directory of cityscapes')
args = parser.parse_args()
main(args)
'''
