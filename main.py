"""People Counter."""
import json
from pathlib import Path

from imutils.video import FPS

"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
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

import socket
import cv2
import mimetypes
from utils import FileUtil, COCO_LABELS
import paho.mqtt.client as mqtt
from argparse import ArgumentParser
from inference import Network
import sys
import time

# MQTT server environment variables

HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                             "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(
        MQTT_HOST,
        MQTT_PORT,
        MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    model_path = args.model
    input = args.input
    device = args.device

    ### TODO: Load the model through `infer_network` ###
    input_dict = infer_network.load_model(model_path, device)

    ### TODO: Handle the input stream ###
    # camera
    input_type = None
    if input == "cam":
        input_type = "cam"
    elif isinstance(input, str) and Path(input).is_file():
        mimetype = mimetypes.guess_type(input)[0]
        if mimetype:
            mimetype = mimetype.split('/')[0]
            if mimetype == 'video':
                input_type = "video"
            elif mimetype == "image":
                input_type = "image"

    if input_type not in ["cam", "video", "image"]:
        raise Exception("Invalid input parameter")

    # handler different inputs
    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(input)
    cap.open(input)
    w = int(cap.get(3))
    h = int(cap.get(4))
    request_id = 0

    # counters
    people_total_counter = 0
    people_last_counter = 0
    frames_counter = 0
    # this value can be adjusted according to the model accuracy
    frames_interval_baseline = 50
    start_time = 0
    # labels map
    labels_map = COCO_LABELS
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        ### TODO: Pre-process the image as needed ###
        net_input = infer_network.process_input(frame)
        net_input_dict = {'image_tensor': net_input}
        if "image_info" in input_dict:
            net_input_dict['image_info'] = net_input.shape[1:]
        ### TODO: Start asynchronous inference for specified request ###
        infer_network.async_exec_net(net_input_dict, request_id)
        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            ### TODO: Get the results of the inference request ###
            output_dict = infer_network.get_output()
            ### TODO: Extract any desired stats from the results ###
            predictions = infer_network.process_output(output_dict, target_size=(h, w), boxes_threshold=0.3)
            people_curr_count = 0
            for box in predictions:
                label_id = box.label_id
                label = labels_map[label_id]
                # counting the number of people in the frame
                if label == "person":
                    people_curr_count += 1
                    box.draw(frame)
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            client.publish("person", json.dumps({"count": people_curr_count}))
            frames_counter += 1
            # detect changes between n(frames_interval_baseline) frames
            if frames_counter == frames_interval_baseline:
                # if was a change between on frame to the other, in this case
                # if a new person enters to the scene
                if people_curr_count > people_last_counter:
                    start_time = time.time()
                    # increase the total people counter and send the new value to the gui
                    people_total_counter = people_total_counter + people_curr_count - people_last_counter
                    client.publish("person", json.dumps({"total": people_total_counter}))
                # Person duration in the video is calculated
                if people_curr_count < people_last_counter:
                    time_delta = int(time.time() - start_time)
                    client.publish("person/duration", json.dumps({"duration": time_delta}))

                # update the counter
                people_last_counter = people_curr_count
            frames_counter = frames_counter % frames_interval_baseline  # reset the frame counter
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        cv2.imshow("Frame", frame)
        ### TODO: Write an output image if `single_image_mode` ###
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if input_type == "image":
            cv2.imwrite("out.jpg", frame)
    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
