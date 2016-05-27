#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

from tools import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from six.moves import xrange
import numpy as np
import caffe, os, sys, cv2

CLASSES = ('__background__','license')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

CONF_THRESH = 0.9
NMS_THRESH = 0.3

cfg.TEST.HAS_RPN = True  # Use RPN for proposals
cfg.net = 'zf'
cfg.cpu_mode = True

prototxt = os.path.join(cfg.MODELS_DIR, NETS[cfg.net][0],
                        'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
caffemodel = os.path.join(cfg.DATA_DIR, 'DeepPR_models',
                            NETS[cfg.net][1])

if not os.path.isfile(caffemodel):
    raise IOError('{:s} not found.'.format(caffemodel))

if cfg.cpu_mode:
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()
    caffe.set_device(cfg.gpu_id)

net = caffe.Net(prototxt, caffemodel, caffe.TEST)

# Warm up
for i in xrange(2):
    _, _ = im_detect(net, 128 * np.ones((300, 500, 3), dtype=np.uint8))

def detect(image):
    regions = {}
    scores, boxes = im_detect(net, image)

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

        get_region = lambda bbox : image[bbox[1]:bbox[3], bbox[0]: bbox[2], :]

        regions[cls] = [(get_region(dets[i, :4]), dets[i, -1]) for i in inds]

    return regions

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    if len(sys.argv) != 2:
        print('Usage: python detector.py image')
        sys.exit()

    regions = detect(cv2.imread(sys.argv[1]))

    for cls in regions:
        print('%s:' % cls)

        print(regions[cls])
        for r, s in regions[cls]:
            plt.imshow(r[:,:,::-1])
            print('\tscore = %f' % s)

    plt.show()
