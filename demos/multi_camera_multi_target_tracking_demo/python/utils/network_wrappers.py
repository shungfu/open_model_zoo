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

import json
import logging as log
from abc import ABC, abstractmethod
from types import SimpleNamespace as namespace

import cv2
import numpy as np

from utils.ie_tools import IEModel
from .segm_postprocess import postprocess
from utils.yolov8 import yolov8_preprocess_image, yolov8_postprocess, image_to_tensor


class DetectorInterface(ABC):
    @abstractmethod
    def run_async(self, frames, index):
        pass

    @abstractmethod
    def wait_and_grab(self):
        pass


class Detector(IEModel, DetectorInterface):
    """Wrapper class for detector"""

    def __init__(self, core, model_path, trg_classes, conf=.6,
                 device='CPU', max_num_frames=1):
        super().__init__(core, model_path, device, 'Object Detection', max_num_frames)

        self.trg_classes = trg_classes
        self.confidence = conf
        self.expand_ratio = (1., 1.)
        self.max_num_frames = max_num_frames

    def run_async(self, frames, index):
        assert len(frames) <= self.max_num_frames
        self.shapes = []
        for id, frame in enumerate(frames):
            self.shapes.append(frame.shape)
            self.forward_async(frame, id)

    def wait_and_grab(self, only_target_class=True):
        all_detections = []
        outputs = self.grab_all_async()
        for i, out in enumerate(outputs):
            detections = self.__decode_detections(out, self.shapes[i], only_target_class)
            all_detections.append(detections)
        return all_detections

    def get_detections(self, frames):
        """Returns all detections on frames"""
        self.run_async(frames)
        return self.wait_and_grab()

    def __decode_detections(self, out, frame_shape, only_target_class):
        """Decodes raw SSD output"""
        detections = []

        for detection in out[0, 0]:
            if only_target_class and detection[1] not in self.trg_classes:
                continue

            confidence = detection[2]
            if confidence < self.confidence:
                continue

            left = int(max(detection[3], 0) * frame_shape[1])
            top = int(max(detection[4], 0) * frame_shape[0])
            right = int(min(detection[5], 1) * frame_shape[1])
            bottom = int(min(detection[6], 1) * frame_shape[0])
            if self.expand_ratio != (1., 1.):
                w = (right - left)
                h = (bottom - top)
                dw = w * (self.expand_ratio[0] - 1.) / 2
                dh = h * (self.expand_ratio[1] - 1.) / 2
                left = max(int(left - dw), 0)
                right = int(right + dw)
                top = max(int(top - dh), 0)
                bottom = int(bottom + dh)

            detections.append(((left, top, right, bottom), confidence))

        if len(detections) > 1:
            detections.sort(key=lambda x: x[1], reverse=True)

        return detections


class VectorCNN(IEModel):
    """Wrapper class for a network returning a vector"""

    def __init__(self, core, model_path, device='CPU', max_reqs=100):
        self.max_reqs = max_reqs
        super().__init__(core, model_path, device, 'Object Reidentification', self.max_reqs)

    def forward(self, batch):
        """Performs forward of the underlying network on a given batch"""
        assert len(batch) <= self.max_reqs
        for id, frame in enumerate(batch):
            super().forward_async(frame, id)
        outputs = self.grab_all_async()
        return outputs

    def forward_async(self, batch):
        """Performs async forward of the underlying network on a given batch"""
        assert len(batch) <= self.max_reqs
        for frame in batch:
            super().forward_async(frame)

    def wait_and_grab(self):
        outputs = self.grab_all_async()
        return outputs


class MaskRCNN(IEModel, DetectorInterface):
    """Wrapper class for a network returning masks of objects"""

    def __init__(self, core, model_path, trg_classes, conf=.6,
                 device='CPU', max_reqs=100):
        self.trg_classes = trg_classes
        self.max_reqs = max_reqs
        self.confidence = conf
        super().__init__(core, model_path, device, 'Instance Segmentation', self.max_reqs)

        self.input_keys = {'image'}
        self.output_keys = {'boxes', 'labels', 'masks'}
        self.input_keys_segmentoly = {'im_info', 'im_data'}
        self.output_keys_segmentoly = {'boxes', 'scores', 'classes', 'raw_masks'}

        self.segmentoly_type = self.check_segmentoly_type()
        self.input_tensor_name = 'im_data' if self.segmentoly_type else 'image'
        self.n, self.c, self.h, self.w = self.model.input(self.input_tensor_name).shape

    def check_segmentoly_type(self):
        for input_tensor_name in self.input_keys_segmentoly:
            try:
                self.model.input(input_tensor_name)
            except RuntimeError:
                return False
        for output_tensor_name in self.output_keys_segmentoly:
            try:
                self.model.output(output_tensor_name)
            except RuntimeError:
                return False
        return True

    def preprocess(self, frame):
        image_height, image_width = frame.shape[:2]
        scale = min(self.h / image_height, self.w / image_width)
        processed_image = cv2.resize(frame, None, fx=scale, fy=scale)
        processed_image = processed_image.astype('float32').transpose(2, 0, 1)

        return namespace(
            original_image=frame,
            meta=namespace(
                original_size=frame.shape[:2],
                processed_size=processed_image.shape[1:3],
            ),
            im_data=processed_image,
            im_info=np.array([processed_image.shape[1], processed_image.shape[2], 1.0], dtype='float32'),
        )

    def forward(self, im_data, im_info):
        if (self.h - im_data.shape[1] < 0) or (self.w - im_data.shape[2] < 0):
            raise ValueError('Input image should resolution of {}x{} or less, '
                             'got {}x{}.'.format(self.w, self.h, im_data.shape[2], im_data.shape[1]))
        im_data = np.pad(im_data, ((0, 0),
                                   (0, self.h - im_data.shape[1]),
                                   (0, self.w - im_data.shape[2])),
                         mode='constant', constant_values=0).reshape(1, self.c, self.h, self.w)
        feed_dict = {self.input_tensor_name: im_data}
        if im_info is not None:
            im_info = im_info.reshape(1, *im_info.shape)
            feed_dict['im_info'] = im_info
        self.infer_queue[0].infer(feed_dict)
        if self.segmentoly_type:
            output = {name: self.infer_queue[0].get_tensor(name).data for name in self.output_keys_segmentoly}
            valid_detections_mask = output['classes'] > 0
            classes = output['classes'][valid_detections_mask]
            boxes = output['boxes'][valid_detections_mask]
            scores = output['scores'][valid_detections_mask]
            masks = output['raw_masks'][valid_detections_mask]
        else:
            output = {name: self.infer_queue[0].get_tensor(name).data for name in self.output_keys}
            valid_detections_mask = np.sum(output['boxes'], axis=1) > 0
            classes = output['labels'][valid_detections_mask] + 1
            boxes = output['boxes'][valid_detections_mask][:, :4]
            scores = output['boxes'][valid_detections_mask][:, 4]
            masks = output['masks'][valid_detections_mask]
        return boxes, classes, scores, np.full(len(classes), 0, dtype=np.int32), masks

    def get_detections(self, frames, return_cropped_masks=True, only_target_class=True):
        outputs = []
        for frame in frames:
            data_batch = self.preprocess(frame)
            im_data = data_batch.im_data
            im_info = data_batch.im_info if self.segmentoly_type else None
            meta = data_batch.meta

            boxes, classes, scores, _, masks = self.forward(im_data, im_info)
            scores, classes, boxes, masks = postprocess(scores, classes, boxes, masks,
                                                        im_h=meta.original_size[0],
                                                        im_w=meta.original_size[1],
                                                        im_scale_y=meta.processed_size[0] / meta.original_size[0],
                                                        im_scale_x=meta.processed_size[1] / meta.original_size[1],
                                                        full_image_masks=True, encode_masks=False,
                                                        confidence_threshold=self.confidence,
                                                        segmentoly_type=self.segmentoly_type)
            frame_output = []
            for i in range(len(scores)):
                if only_target_class and classes[i] not in self.trg_classes:
                    continue

                bbox = [int(value) for value in boxes[i]]
                if return_cropped_masks:
                    left, top, right, bottom = bbox
                    mask = masks[i][top:bottom, left:right]
                else:
                    mask = masks[i]

                frame_output.append([bbox, scores[i], mask])

            outputs.append(frame_output)

        return outputs

    def run_async(self, frames, index):
        self.frames = frames

    def wait_and_grab(self):
        return self.get_detections(self.frames)


class DetectionsFromFileReader(DetectorInterface):
    """Read detection from *.json file.
    Format of the file should be:
    [
        {'frame_id': N,
         'scores': [score0, score1, ...],
         'boxes': [[x0, y0, x1, y1], [x0, y0, x1, y1], ...]},
        ...
    ]
    """

    def __init__(self, input_file, score_thresh):
        self.input_file = input_file
        self.score_thresh = score_thresh
        self.detections = []
        log.info('Loading {}'.format(input_file))
        with open(input_file) as f:
            all_detections = json.load(f)
        for source_detections in all_detections:
            detections_dict = {}
            for det in source_detections:
                detections_dict[det['frame_id']] = {'boxes': det['boxes'], 'scores': det['scores']}
            self.detections.append(detections_dict)

    def run_async(self, frames, index):
        self.last_index = index

    def wait_and_grab(self):
        output = []
        for source in self.detections:
            valid_detections = []
            if self.last_index in source:
                for bbox, score in zip(source[self.last_index]['boxes'], source[self.last_index]['scores']):
                    if score > self.score_thresh:
                        bbox = [int(value) for value in bbox]
                        valid_detections.append((bbox, score))
            output.append(valid_detections)
        return output
from openvino.runtime import AsyncInferQueue, Model
class YOLOv8Seg(DetectorInterface):
    """Wrapper class for a network returning masks of objects"""

    def __init__(self, core, model_path, trg_classes, conf=.6,
                 device='CPU', max_reqs=100):
        self.trg_classes = 80 #trg_classes
        self.max_reqs = max_reqs
        self.confidence = conf
        self.load_model(core, model_path, device, 'YOLOv8 Instance Segmentation', self.max_reqs)
        self.outputs = {}
        self.output_keys = {'output0', 'output1'}

    def _preprocess(self, img):
        preprocessed_image = yolov8_preprocess_image(img)
        input_tensor = image_to_tensor(preprocessed_image)
        return input_tensor

    def completion_callback(self, infer_request, id):
        #self.outputs[id] = infer_request.get_tensor(self.output_tensor_name).data[:]
        self.outputs[id] = {name: infer_request.get_tensor(name).data[:] for name in self.output_keys}

    def forward(self, img):
        """Performs forward pass of the wrapped model"""
        self.forward_async(img, 0)
        self.infer_queue.wait_all()
        return self.outputs.pop(0)

    def forward_async(self, img, req_id):
        input_data = {self.input_tensor_name: self._preprocess(img)}
        self.infer_queue.start_async(input_data, req_id)

    def grab_all_async(self):
        self.infer_queue.wait_all()
        return [self.outputs.pop(i) for i in range(len(self.outputs))]

    def get_allowed_inputs_len(self):
        return (1, 2)

    def get_allowed_outputs_len(self):
        return (1, 2, 3, 4, 5)

    def get_input_shape(self):
        """Returns an input shape of the wrapped model"""
        return self.model.inputs[0].shape

    def load_model(self, core, model_path, device, model_type, num_reqs=1):
        """Loads a model in the OpenVINO Runtime format"""

        log.info('Reading {} model {}'.format(model_type, model_path))
        self.model = core.read_model(model_path)

#self.model.inputs:[<Output: names[images] shape[1,3,?,?] type: f32>]
#self.model.outputs:[<Output: names[output0] shape[1,116,3..] type: f32>, <Output: names[output1] shape[1,32,1..,1..] type: f32>]

        if len(self.model.inputs) not in self.get_allowed_inputs_len():
            raise RuntimeError("Supports topologies with only {} inputs, but got {}"
                .format(self.get_allowed_inputs_len(), len(self.model.inputs)))
        if len(self.model.outputs) not in self.get_allowed_outputs_len():
            raise RuntimeError("Supports topologies with only {} outputs, but got {}"
                .format(self.get_allowed_outputs_len(), len(self.model.outputs)))

        self.input_tensor_name = self.model.inputs[0].get_any_name()
        self.output_tensor_name = self.model.outputs[0].get_any_name()
        # Loading model to the plugin
        if device != "CPU":
            self.model.reshape({0: [1, 3, 640, 640]})
        self.compiled_model = core.compile_model(self.model, device)

#self.compiled_model.inputs:[<ConstOutput: names[images] shape[1,3,640,640] type: f32>]
#self.compiled_model.outputs:[<ConstOutput: names[output0] shape[1,116,8400] type: f32>, <ConstOutput: names[output1] shape[1,32,160,160] type: f32>]

        self.infer_queue = AsyncInferQueue(self.compiled_model, num_reqs)
        self.infer_queue.set_callback(self.completion_callback)
        log.info('The {} model {} is loaded to {}'.format(model_type, model_path, device))

    def run_async(self, frames, index):
        #assert len(frames) <= self.max_num_frames
        self.shapes = []
        self.frame = []
        for id, frame in enumerate(frames):
            self.shapes.append(frame.shape)
            self.frame.append(frame)
            self.forward_async(frame, id)

    def wait_and_grab(self, only_target_class=True):
        all_detections = []
        outputs = self.grab_all_async()
        for i, out in enumerate(outputs):
            detections = self.__decode_detections(out, self.shapes[i], self.frame[i], only_target_class)
            all_detections.append(detections)
        return all_detections

    def get_detections(self, frames):
        """Returns all detections on frames"""
        self.run_async(frames)
        return self.wait_and_grab()

    def __decode_detections(self, out, frame_shape, frame, only_target_class):
        detections = []
        boxes = out['output0']
        # print("boxes:{}".format(boxes))
        masks = out['output1']
        # print("masks:{}".format(masks))
        input_hw = [640,640] #input_tensor.shape[2:]
        detections = yolov8_postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=frame, pred_masks=masks)
        # print("detections:{}".format(detections[0]))

        return detections
