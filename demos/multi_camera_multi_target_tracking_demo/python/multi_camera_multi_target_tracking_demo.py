#!/usr/bin/env python3

"""
 Copyright (c) 2019-2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import argparse
import time
import queue
from threading import Thread
import json
import logging as log
import os
from pathlib import Path
import random
import sys

import cv2 as cv

from utils.network_wrappers import Detector, VectorCNN, MaskRCNN, DetectionsFromFileReader, YOLOv8Seg
from mc_tracker.mct import MultiCameraTracker
from utils.analyzer import save_embeddings
from utils.misc import read_py_config, check_pressed_keys
from utils.video import MulticamCapture, NormalizerCLAHE
from utils.visualization import visualize_multicam_detections, get_target_size
from openvino.runtime import Core, get_version


sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/openvino/model_zoo'))

import monitors
from model_api.performance_metrics import PerformanceMetrics

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def check_detectors(args):
    detectors = {
        '--m_detector': args.m_detector,
        '--m_segmentation': args.m_segmentation,
        '--detections': args.detections
    }
    non_empty_detectors = [(det, value) for det, value in detectors.items() if value]
    det_number = len(non_empty_detectors)
    if det_number == 0:
        log.error('No detector specified, please specify one of the following parameters: '
                  '\'--m_detector\', \'--m_segmentation\' or \'--detections\'')
    elif det_number > 1:
        det_string = ''.join('\n\t{}={}'.format(det[0], det[1]) for det in non_empty_detectors)
        log.error('Only one detector expected but got {}, please specify one of them:{}'
                  .format(len(non_empty_detectors), det_string))
    return det_number


def update_detections(output, detections, frame_number):
    for i, detection in enumerate(detections):
        entry = {'frame_id': frame_number, 'scores': [], 'boxes': []}
        for det in detection:
            entry['boxes'].append(det[0])
            entry['scores'].append(float(det[1]))
        output[i].append(entry)


def save_json_file(save_path, data, description=''):
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_path, 'w') as outfile:
        json.dump(data, outfile)
    if description:
        log.debug('{} saved to {}'.format(description, save_path))


class FramesThreadBody:
    def __init__(self, capture, max_queue_length=2):
        self.process = True
        self.frames_queue = queue.Queue()
        self.capture = capture
        self.max_queue_length = max_queue_length

    def __call__(self):
        while self.process:
            if self.frames_queue.qsize() > self.max_queue_length:
                #print("self.frames_queue.qsize():{}".format(self.frames_queue.qsize()))
                time.sleep(0.1)
                continue
            has_frames, frames = self.capture.get_frames()
            if not has_frames and self.frames_queue.empty():
                self.process = False
                break
            if has_frames:
                self.frames_queue.put(frames)


def run(params, config, capture, detector, reid):
    win_name = 'Multi camera tracking'
    cv.namedWindow(win_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED)
    cv.resizeWindow(win_name, 1920, 1080)
    #cv.namedWindow('mask', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_EXPANDED)
    frame_number = 0
    output_detections = [[] for _ in range(capture.get_num_sources())]
    key = -1

    if config.normalizer_config.enabled:
        capture.add_transform(
            NormalizerCLAHE(
                config.normalizer_config.clip_limit,
                config.normalizer_config.tile_size,
            )
        )

    tracker = MultiCameraTracker(capture.get_num_sources(), reid, config.sct_config,
                                 **vars(config.mct_config), visual_analyze=config.analyzer)

    thread_body = FramesThreadBody(capture, max_queue_length=len(capture.captures) * 30)
    frames_thread = Thread(target=thread_body)
    frames_thread.start()

    frames_read = False
    set_output_params = False

    prev_frames = thread_body.frames_queue.get()
    detector.run_async(prev_frames, frame_number)
    metrics = PerformanceMetrics()
    presenter = monitors.Presenter(params.utilization_monitors, 0)

    while thread_body.process:
        if not params.no_show:
            key = check_pressed_keys(key)
            if key == 27:
                break
            presenter.handleKey(key)
        start_time = time.perf_counter()
        try:
            frames = thread_body.frames_queue.get_nowait()
            frames_read = True
        except queue.Empty:
            frames = None

        if frames is None:
            continue

        all_detections = detector.wait_and_grab()
        all_detections_yolov8 = all_detections.copy()
        if params.save_detections:
            update_detections(output_detections, all_detections, frame_number)
        frame_number += 1
        detector.run_async(frames, frame_number)

        all_masks = [[] for _ in range(len(all_detections))]
        for i, detections in enumerate(all_detections):
            #all_detections[i] = [det[0] for det in detections]
            #all_masks[i] = [det[2] for det in detections if len(det) == 3]
            # YOLOv8
            #[print(det['det']) for det in detections]
            for dets in detections:
                all_detections[i] = [list(det[:4].astype(int)) for det in dets['det']]
                #all_masks[i] = [det for det in dets['segment']]
                
        tracker.process(prev_frames, all_detections, all_masks)
        tracked_objects = tracker.get_tracked_objects()
        #tracked_objects = [[]] #[[],[],[],[]]


        vis = visualize_multicam_detections(prev_frames, tracked_objects, all_detections_yolov8,
                                            **vars(config.visualization_config))
        metrics.update(start_time, vis)
        presenter.drawGraphs(vis)
        if not params.no_show:
            cv.imshow(win_name, vis)

        if frames_read and not set_output_params:
            set_output_params = True
            if len(params.output_video):
                frame_size = [frame.shape[::-1] for frame in frames]
                fps = capture.get_fps()
                target_width, target_height = get_target_size(
                    frame_size, None, **vars(config.visualization_config))
                video_output_size = (target_width, target_height)
                fourcc = cv.VideoWriter_fourcc(*'XVID')
                output_video = cv.VideoWriter(params.output_video, fourcc, min(fps), video_output_size)
            else:
                output_video = None
        if set_output_params and output_video:
            output_video.write(cv.resize(vis, video_output_size))

        prev_frames, frames = frames, prev_frames

    metrics.log_total()
    for rep in presenter.reportMeans():
        log.info(rep)

    thread_body.process = False
    frames_thread.join()

    if len(params.history_file):
        save_json_file(params.history_file, tracker.get_all_tracks_history(), description='History file')
    if len(params.save_detections):
        save_json_file(params.save_detections, output_detections, description='Detections')

    if len(config.embeddings.save_path):
        save_embeddings(tracker.scts, **vars(config.embeddings))
    detector.infer_queue.wait_all()


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    """Prepares data for the object tracking demo"""
    parser = argparse.ArgumentParser(description='Multi camera multi object \
                                                  tracking live demo script')
    parser.add_argument('-i', '--input', required=True, nargs='+',
                        help='Required. Input sources (indexes of cameras or paths to video files)')
    parser.add_argument('--loop', default=False, action='store_true',
                        help='Optional. Enable reading the input in a loop')
    parser.add_argument('--config', type=str, default=os.path.join(current_dir, 'configs/person.py'), required=False,
                        help='Configuration file')

    parser.add_argument('--detections', type=str, help='JSON file with bounding boxes')

    parser.add_argument('-m', '--m_detector', type=str, required=False,
                        help='Path to the object detection model')
    parser.add_argument('--t_detector', type=float, default=0.6,
                        help='Threshold for the object detection model')

    parser.add_argument('--m_segmentation', type=str, required=False,
                        help='Path to the object instance segmentation model')
    parser.add_argument('--t_segmentation', type=float, default=0.6,
                        help='Threshold for object instance segmentation model')

    parser.add_argument('--m_yolov8seg', type=str, required=False,
                        help='Path to the YOLOv8 object instance segmentation model')
    parser.add_argument('--m_reid', type=str, required=False,
                        help='Required. Path to the object re-identification model')

    parser.add_argument('--output_video', type=str, default='', required=False,
                        help='Optional. Path to output video')
    parser.add_argument('--history_file', type=str, default='', required=False,
                        help='Optional. Path to file in JSON format to save results of the demo')
    parser.add_argument('--save_detections', type=str, default='', required=False,
                        help='Optional. Path to file in JSON format to save bounding boxes')
    parser.add_argument("--no_show", help="Optional. Don't show output", action='store_true')

    parser.add_argument('-d', '--device', type=str, default='CPU')
    parser.add_argument('-u', '--utilization_monitors', default='', type=str,
                        help='Optional. List of monitors to show initially.')

    args = parser.parse_args()
    if check_detectors(args) != 1:
        sys.exit(1)

    if len(args.config):
        log.debug('Reading config from {}'.format(args.config))
        config = read_py_config(args.config)
    else:
        log.error('No configuration file specified. Please specify parameter \'--config\'')
        sys.exit(1)

    random.seed(config.random_seed)
    capture = MulticamCapture(args.input, args.loop)

    log.info('OpenVINO Runtime')
    log.info('\tbuild: {}'.format(get_version()))
    core = Core()

    if args.detections:
        print("==========DETECTIONS==========")
        object_detector = DetectionsFromFileReader(args.detections, args.t_detector)
    elif args.m_segmentation:
        print("==========SEGMENTATION==========")
        object_detector = MaskRCNN(core, args.m_segmentation,
                                   config.obj_segm.trg_classes,
                                   args.t_segmentation,
                                   args.device,
                                   capture.get_num_sources())
    else:
        print("==========YOLOV8-SEG==========")
        # object_detector = Detector(core, args.m_detector,
        #                            config.obj_det.trg_classes,
        #                            args.t_detector,
        #                            args.device,
        #                            capture.get_num_sources())

        object_detector_yolov8 = YOLOv8Seg(core, args.m_yolov8seg,
                                   config.obj_det.trg_classes,
                                   args.t_detector,
                                   args.device,
                                   16) #capture.get_num_sources())

        object_detector = object_detector_yolov8

    if args.m_reid:
        print("==========VECTORCNN==========")
        object_recognizer = VectorCNN(core, args.m_reid, args.device)
    else:
        print("==========NONE==========")
        object_recognizer = None

    run(args, config, capture, object_detector, object_recognizer)


if __name__ == '__main__':
    sys.exit(main() or 0)
