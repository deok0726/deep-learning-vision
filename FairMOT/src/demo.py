from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq
import sys

logger.setLevel(logging.INFO)


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    # use opt argument later
    opt.input_video = '../videos/NY.mp4'
    video_route = osp.splitext(osp.abspath(opt.input_video))[0]
    video_name = osp.basename(video_route)
    opt.output_root = '../demos' + '_' + video_name
    result_root = opt.output_root

    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    eval_seq(opt, dataloader, 'mot', result_filename,
             save_dir=frame_dir, show_image=False, frame_rate=frame_rate,
             use_cuda=opt.gpus!=[-1])

    if opt.output_format == 'video':
        # output_video_path = osp.join(result_root, osp.join(osp.basename(osp.splitext(osp.abspath(opt.input_video))[0]),'results.mp4'))
        result_name = video_route + '_results.mp4'
        output_name = osp.basename(result_name)
        output_video_path = osp.join(result_root, output_name)
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
        os.system(cmd_str)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    opt = opts().init()
    demo(opt)
