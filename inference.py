#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import numpy as np
import cv2
import logging as log
from pathlib import Path
from box import Box
from utils import ModelZoo
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.ie = IECore()
        self.net = None
        self.exec_net = None
        self.current_request = None

    def load_model(self, model, device):
        model_xml = model if isinstance(model, Path) else Path(model)
        model_bin = model_xml.parent.joinpath("{}.bin".format(model_xml.stem))
        assert model_xml.exists(), "model xml file not found"
        assert model_bin.exists(), "model .bin file not found"
        self.net = self.ie.read_network(model=str(model_xml), weights=str(model_bin))
        self.exec_net = self.ie.load_network(network=self.net, device_name=device)
        return {k: self.get_input_shape(k) for k in self.net.inputs.keys()}

    def get_input_shape(self, layer_name):
        ### TODO: Return the shape of the input layer ###
        return self.net.inputs[layer_name].shape


    def async_exec_net(self, net_input, request_id):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        self.current_request = self.exec_net.start_async(request_id, inputs=net_input)

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        status = self.current_request.wait()
        return status

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.current_request.outputs

    def process_input(self, input_image):
        _, c, h, w = self.get_input_shape("image_tensor")
        image = np.copy(input_image)
        image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, c, w, h)
        return image

    def process_output(self, output_dict: dict, target_size: tuple, boxes_threshold=0.5, masks_threshold=0.1):
        h, w = target_size
        result = []
        if "masks" in output_dict:
            masks = output_dict["masks"]
            boxes = output_dict["reshape_do_2d"]
            for idx, (box, mask) in enumerate(zip(boxes, masks)):
                class_id = np.int(box[1])
                confidence = box[2]
                if confidence > boxes_threshold:
                    xmin, ymin, xmax, ymax = box[3:]
                    ymin, xmin, ymax, xmax = int(ymin * h), int(xmin * w), int(ymax * h), int(xmax * w)
                    bbox_height, bbox_width = ymax - ymin, xmax - xmin
                    detection_mask = mask[idx]  # get mask
                    detection_mask = cv2.resize(detection_mask, (bbox_width, bbox_height),
                                                interpolation=cv2.INTER_CUBIC)
                    result_mask = np.zeros((h, w))
                    result_mask[ymin:ymax, xmin:xmax] = detection_mask
                    result_mask[np.where(result_mask > 0.1)] = 1
                    mask = result_mask.astype(np.uint8)
                    result.append(Box(xmin, ymin, xmax, ymax, confidence, class_id, mask))
        else:
            output_name, output_info = "", ""
            # fetching output layer name
            for output_key in self.net.outputs:
                if self.net.layers[output_key].type == "DetectionOutput":
                    output_name, output_info = output_key, self.net.outputs[output_key]
            # obtaining the predicted boxes
            predicted_boxes = output_dict[output_name][0, 0, :, :]
            for box in predicted_boxes:
                class_id = np.int(box[1])
                confidence = box[2]
                if confidence > boxes_threshold:
                    xmin, ymin, xmax, ymax = box[3:]
                    ymin, xmin, ymax, xmax = int(ymin * h), int(xmin * w), int(ymax * h), int(xmax * w)
                    result.append(Box(xmin, ymin, xmax, ymax, confidence, class_id))
        return result


if __name__ == '__main__':
    #ModelZoo.download("ssd_mobilenet_v2_coco", model_folder="./models", device="CPU",force_download=True)
    #ModelZoo.download("faster_rcnn_inception_v2_coco", model_folder="./models", device="CPU", force_download=True)
    ModelZoo.download("mask_rcnn_inception_v2_coco", model_folder="./models", device="CPU", force_download=True)

