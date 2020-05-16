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
