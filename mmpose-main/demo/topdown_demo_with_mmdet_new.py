# Copyright (c) OpenMMLab. All rights reserved.
import logging
import mimetypes
import os
import time
import argparse
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
from mmengine.logging import print_log

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    det_result = inference_detector(detector, img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                   pred_instance.scores > args.bbox_thr)]
    bboxes = bboxes[nms(bboxes, args.nms_thr), :4]
    print("bboxes:",bboxes)
    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None),bboxes



class mmpose():
    def __init__(self,args_input=None):
        
        """
        args_input输入为字典形式
        Visualize the demo images.

        Using mmdet to detect the human.
        """
        parser = ArgumentParser()
        parser.add_argument('--det_config', help='Config file for detection')
        parser.add_argument('--det_checkpoint', help='Checkpoint file for detection')
        parser.add_argument('--pose_config', help='Config file for pose')
        parser.add_argument('--pose_checkpoint', help='Checkpoint file for pose')
        parser.add_argument(
            '--input', type=str, default='', help='Image/Video file')
        parser.add_argument(
            '--show',
            action='store_true',
            default=False,
            help='whether to show img')
        parser.add_argument(
            '--output',
            type=str,
            default='',
        )
        parser.add_argument(
            '--output-root',
            type=str,
            default='',
            help='root of the output img file. '
            'Default not saving the visualization images.')
        parser.add_argument(
            '--save-predictions',
            action='store_true',
            default=False,
            help='whether to save predicted results')
        parser.add_argument(
            '--device', default='cuda:0', help='Device used for inference')
        parser.add_argument(
            '--det-cat-id',
            type=int,
            default=0,
            help='Category id for bounding box detection model')
        parser.add_argument(
            '--bbox-thr',
            type=float,
            default=0.3,
            help='Bounding box score threshold')
        parser.add_argument(
            '--nms-thr',
            type=float,
            default=0.3,
            help='IoU threshold for bounding box NMS')
        parser.add_argument(
            '--kpt-thr',
            type=float,
            default=0.3,
            help='Visualizing keypoint thresholds')
        parser.add_argument(
            '--draw-heatmap',
            action='store_true',
            default=False,
            help='Draw heatmap predicted by the model')
        parser.add_argument(
            '--show-kpt-idx',
            action='store_true',
            default=False,
            help='Whether to show the index of keypoints')
        parser.add_argument(
            '--skeleton-style',
            default='mmpose',
            type=str,
            choices=['mmpose', 'openpose'],
            help='Skeleton style selection')
        parser.add_argument(
            '--radius',
            type=int,
            default=4,
            help='Keypoint radius for visualization')
        parser.add_argument(
            '--thickness',
            type=int,
            default=2,
            help='Link thickness for visualization')
        parser.add_argument(
            '--show-interval', type=int, default=0, help='Sleep seconds per frame')
        parser.add_argument(
            '--alpha', type=float, default=0.8, help='The transparency of bboxes')
        
        parser.add_argument(
            '--draw-bbox', action='store_true', help='Draw bboxes of instances')
        
        #允许输入合并
        cli_args = parser.parse_args()
        if args_input is not None:
            # 如果传入的是字典，转换为Namespace
            if isinstance(args_input, dict):
                args_input = argparse.Namespace(**args_input)
            
            # 遍历调用参数，覆盖命令行参数
            for key, value in vars(args_input).items():
                if hasattr(cli_args, key):
                    setattr(cli_args, key, value)
        
        # 使用合并后的参数对象
        self.args = cli_args
        assert has_mmdet, 'Please install mmdet to run the demo.'

        

        # assert self.args.show or (self.args.output_root != '') or (self.args.output != '') 
        # assert self.args.input != ''
        assert self.args.det_config is not None
        assert self.args.det_checkpoint is not None

       

        # build detector
        self.detector = init_detector(
            self.args.det_config, self.args.det_checkpoint, device=self.args.device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

        # build pose estimator
        self.pose_estimator = init_pose_estimator(
            self.args.pose_config,
            self.args.pose_checkpoint,
            device=self.args.device,
            cfg_options=dict(
                model=dict(test_cfg=dict(output_heatmaps=self.args.draw_heatmap))))

        # build visualizer
        self.pose_estimator.cfg.visualizer.radius = self.args.radius
        self.pose_estimator.cfg.visualizer.alpha = self.args.alpha
        self.pose_estimator.cfg.visualizer.line_width = self.args.thickness
        self.visualizer = VISUALIZERS.build(self.pose_estimator.cfg.visualizer)
        # the dataset_meta is loaded from the checkpoint and
        # then pass to the model in init_pose_estimator
        self.visualizer.set_dataset_meta(
            self.pose_estimator.dataset_meta, skeleton_style=self.args.skeleton_style)


            
    #输出结果
    def run(self,input,output):
        #判断输入类型
        if self.args.input == 'webcam':
            self.input_type = 'webcam'
        else:
            self.input_type = mimetypes.guess_type(input)[0].split('/')[0]
        self.output_file = None
        
        if self.args.output_root:
            mmengine.mkdir_or_exist(self.args.output_root)
            self.output_file = os.path.join(self.args.output_root,
                                    os.path.basename(self.args.input))
            if self.args.input == 'webcam':
                self.output_file += '.mp4'

        if self.args.save_predictions:
            assert self.args.output_root != ''
            self.args.pred_save_path = f'{self.args.output_root}/results_' \
                f'{os.path.splitext(os.path.basename(input))[0]}.json'
        if self.input_type == 'image':

            # inference
            pred_instances,bboxes = process_one_image(self.args, input, self.detector,
                                            self.pose_estimator, self.visualizer)

            if self.args.save_predictions:
                pred_instances_list = split_instances(pred_instances)

            if self.output_file:
                img_vis = self.visualizer.get_image()
                mmcv.imwrite(mmcv.rgb2bgr(img_vis), self.output_file)
            if output:
                img_vis = self.visualizer.get_image()
                mmcv.imwrite(mmcv.rgb2bgr(img_vis), output)
                

        elif input_type in ['webcam', 'video']:

            if input == 'webcam':
                cap = cv2.VideoCapture(0)
            else:
                cap = cv2.VideoCapture(input)

            video_writer = None
            pred_instances_list = []
            frame_idx = 0

            while cap.isOpened():
                success, frame = cap.read()
                frame_idx += 1

                if not success:
                    break

                # topdown pose estimation
                pred_instances,bboxes = process_one_image(self.args, frame, self.detector,
                                                self.pose_estimator, self.visualizer,
                                                0.001)

                if self.args.save_predictions:
                    # save prediction results
                    pred_instances_list.append(
                        dict(
                            frame_id=frame_idx,
                            instances=split_instances(pred_instances)))

                # output videos
                if self.output_file:
                    frame_vis = self.visualizer.get_image()

                    if video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        # the size of the image with visualization may vary
                        # depending on the presence of heatmaps
                        video_writer = cv2.VideoWriter(
                            self.output_file,
                            fourcc,
                            25,  # saved fps
                            (frame_vis.shape[1], frame_vis.shape[0]))

                    video_writer.write(mmcv.rgb2bgr(frame_vis))

                if self.args.show:
                    # press ESC to exit
                    if cv2.waitKey(5) & 0xFF == 27:
                        break

                    time.sleep(self.args.show_interval)

            if video_writer:
                video_writer.release()

            cap.release()

        else:
            self.args.save_predictions = False
            raise ValueError(
                f'file {os.path.basename(input)} has invalid format.')

        if self.args.save_predictions:
            with open(self.args.pred_save_path, 'w') as f:
                json.dump(
                    dict(
                        meta_info=self.pose_estimator.dataset_meta,
                        instance_info=pred_instances_list),
                    f,
                    indent='\t')
            print(f'predictions have been saved at {self.args.pred_save_path}')

        if self.output_file:
            input_type = input_type.replace('webcam', 'video')
            print_log(
                f'the output {input_type} has been saved at {self.output_file}',
                logger='current',
                level=logging.INFO)
            
        return pred_instances,bboxes


if __name__ == '__main__':
    mmpose()
