# Project Write-Up

## Explaining Custom Layers

For this experiment, three different models from the TensorFlow object detection model zoo were evaluated, so all of them are compatible out-of-the-box by OpenVINO. For the model conversion the two classes below were implemented:

### OpenVINOUtil

````python
import os
import subprocess
from pathlib import Path
import typing
from decor import pathassert, exception

os.environ["OPENVINO_DIR"] = "/opt/intel/openvino"


class OpenVINOUtil:
    @staticmethod
    @exception
    @pathassert
    def optimize(frozen_model : typing.Union[str, Path],  model_config : typing.Union[str, Path],transformations_config : typing.Union[str, Path], out_folder: typing.Union[str, Path], h=300, w=300, device="CPU"):
        optimizer_script = "deployment_tools/model_optimizer/mo_tf.py"
        optimizer_script = Path(os.environ["OPENVINO_DIR"]).joinpath(optimizer_script)
        data_type = "FP16" if device == "MYRIAD" else "FP32"
        cmd = '''python "{}" 
                --input_model "{}" 
                --transformations_config  "{}" 
                --output_dir  "{}"   
                --data_type   {}                            
                --reverse_input_channels
                --input_shape "[1, {}, {}, 3]"
                --tensorflow_object_detection_api_pipeline_config "{}"                
            ''' \
            .format(optimizer_script,
                    frozen_model,
                    transformations_config,
                    out_folder,
                    data_type,
                    h, w,
                    model_config
                    )
        cmd = " ".join([line.strip() for line in cmd.splitlines()])
        print(subprocess.check_output(cmd, shell=True).decode())
````

### Model zoo

```python

from pathlib import Path

import requests
import markdown
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from decor import exception
import os

from .openvino_util import OpenVINOUtil
from .file_util import FileUtil

os.environ["MODEL_ZOO"] = "https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md"


class ModelZoo:
    @staticmethod
    @exception
    def available_models() -> []:
        try:
            assert FileUtil.internet_on(), "Not internet connection"
            models = {}
            r = requests.get(os.environ["MODEL_ZOO"])
            if r.status_code == 200:
                md = markdown.Markdown()
                html = md.convert(r.text)
                soup = BeautifulSoup(html, "lxml")
                for a in soup.find_all('a', href=True):
                    model_url = a['href']
                    model_name = a.get_text()
                    path = urlparse(model_url).path
                    ext = os.path.splitext(path)[1]
                    if ext == ".gz":
                        models[model_name] = model_url
            return models
        except Exception as e:
            raise Exception("Error listing the models : {}".format(str(e))) from e

    @classmethod
    @exception
    def download(cls, model_name, model_folder, target_size=(300, 300), device="CPU", force_download=False):
        models = ModelZoo.available_models()
        assert model_name in models, "Invalid model name"
        model_folder = Path(model_folder) if isinstance(model_folder, str) else model_folder
        model_folder.mkdir(exist_ok=True)
        model_url = models[model_name]  # grab model url
        output_file = FileUtil.download_file(model_url, model_folder, force=False)  # download pre-trained model
        output_file_name = output_file.name  # get downloaded file name
        pre_trained_model_folder = model_folder.joinpath(output_file_name[:output_file_name.find('.')])
        frozen_model = list(pre_trained_model_folder.rglob("**/frozen_inference_graph.pb"))[0]
        pipeline_model = list(pre_trained_model_folder.rglob("**/pipeline.config"))[0]
        if model_name.startswith("faster"):
            front_openvino_file = "faster_rcnn_support.json"
        elif model_name.startswith("ssd"):
            front_openvino_file = "ssd_v2_support.json"
        elif model_name.startswith("mask"):
            front_openvino_file = "mask_rcnn_support.json"
        elif model_name.startswith("rfcn"):
            front_openvino_file = "rfcn_support.json"
        else:
            raise Exception("model not supported yet")
        xml_file = pre_trained_model_folder.joinpath("frozen_inference_graph.xml")
        bin_file = pre_trained_model_folder.joinpath("frozen_inference_graph.bin")
        front_openvino_file = Path(os.environ["OPENVINO_DIR"]).joinpath(
            r"deployment_tools/model_optimizer/extensions/front/tf/{}".format(front_openvino_file))
        print(front_openvino_file)
        if not os.path.exists(xml_file) or not os.path.exists(bin_file) or force_download:
            OpenVINOUtil.optimize(str(frozen_model), str(pipeline_model), str(front_openvino_file),
                                  pre_trained_model_folder, h=target_size[0], w=target_size[1], device=device)

```


## Comparing Model Performance

For this experiment, three different models were evaluated:
- ssd_mobilenet_v2_coco_2018_03_29

[![Video](https://img.youtube.com/vi/6jV5GTaYTkI/0.jpg)](https://www.youtube.com/watch?v=6jV5GTaYTkI)

- faster_rcnn_inception_v2_coco_2018_01_28

[![Video](https://img.youtube.com/vi/QoMBHSoCDMs/0.jpg)](https://www.youtube.com/watch?v=QoMBHSoCDMs)

- mask_rcnn_inception_v2_coco_2018_01_28

[![Video](https://img.youtube.com/vi/7OoWQ8PV09U/0.jpg)](https://www.youtube.com/watch?v=7OoWQ8PV09U)

The table below shows the comparition between them:

| Model | pre-conversion (GPU) | post-conversion (CPU) |
| ------------- | ------------- | ------------- |
| ssd_mobilenet_v2_coco_2018_03_29  | approx.FPS: 13.61 | approx. FPS: 37.98 |
| faster_rcnn_inception_v2_coco_2018_01_28  | approx. FPS: 8.31 | approx. FPS: 6.73 |
| mask_rcnn_inception_v2_coco_2018_01_28  | approx. FPS: 0.44 |  approx. FPS: 1.87 |




## Assess Model Use Cases

Some of the scenarios where this application could be adapted are: 
- For security purposes: such as control access solutions
- To estimate the time it takes a person to perform certain activities ( to identify possible bottlenecks):  for instance, voting during elections or withing a bank.

## Assess Effects on End User Needs

One of the critical factors on the "People counter" solution is the detection model accuracy, that was identified when the ssd_mobilenet and the faster-RCNN architectures were compared. The SSD model shows a significant increase in the frame rate of the video; nevertheless, given that some times wasn't able to detect the persons well, it affects the stats. On the other hand, with the faster-RCNN model, the accuracy was much better, but the video's speed was compromised. As a strategy to mitigate that problem, a variable called frames_baseline was added, so this control the numbers of frames were changes between frames will be considered, it can be adjusted according to model. This approach shown improvement in the results.

## For running the app:

```commandline
 npm install npm-run-all --save-dev
 npm-run-all --parallel mqtt ui streaming
./launch_video.sh
```

