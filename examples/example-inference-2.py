#!/usr/bin/env python3
"""Inference with freezed graph."""

import argparse
import os
import sys
import time
import glob

import numpy as np
import tensorflow as tf
import tensorlayer as tl

sys.path.append('.')

from openpose_plus.inference.common import measure, plot_humans, read_imgfile, load_graph, get_op
from openpose_plus.inference.estimator import TfPoseEstimator


class TfPoseestimatorLoader(TfPoseEstimator):

    def __init__(self, path_to_freezed, target_size):
        graph = load_graph(path_to_freezed)

        self.target_size = target_size

        parameter_names = [
            'image',
            #'upsample_size',
            #'upsample_heatmat',
            #'tensor_peaks',
            #'upsample_pafmat',
            'outputs/conf',
            'outputs/paf',
        ]

        for name in parameter_names:
            op = get_op(graph, name)
            print(op)

        #(self.tensor_image, self.upsample_size, self.tensor_heatMat_up, self.tensor_peaks, self.tensor_pafMat_up) = [get_op(graph, name) for name in parameter_names]
        (self.tensor_image, self.tensor_heatmap, self.tensor_paf) = [get_op(graph, name) for name in parameter_names]

        self._warm_up(graph)

    def _warm_up(self, graph):
        self.persistent_sess = tf.InteractiveSession(graph=graph)


def inference(path_to_freezed_model, input_files, plot):
    h, w = 368, 432
    e = measure(lambda: TfPoseestimatorLoader(path_to_freezed_model, target_size=(w, h)),
                'create TfPoseestimatorLoader')
    for idx, img_name in enumerate(input_files):
        image = read_imgfile(img_name, w, h)
        humans, heatMap, pafMap = measure(lambda: e.inference(image), 'inference')
        print('got %d humans from %s' % (len(humans), img_name))
        if humans:
            for h in humans:
                print(h)
        if plot:
          plot_humans(image, heatMap, pafMap, humans, '%02d' % (idx + 1))


def parse_args():
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument(
        '--path-to-freezed-model', type=str, default='checkpoints/freezed', help='path to freezed-model', required=True)
    #parser.add_argument('--images', type=str, default='', help='comma separate list of image filenames', required=True)
    parser.add_argument('--images-dir', type=str, default='', help='directory containing images', required=True)
    parser.add_argument('--limit', type=int, default=1000, help='max number of images.')
    parser.add_argument('--plot', type=bool, default=False, help='draw the results')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # image_files = [f for f in args.images.split(',') if f]
    image_files = glob.glob(args.images_dir)[:args.limit]
    inference(args.path_to_freezed_model, image_files, args.plot)
