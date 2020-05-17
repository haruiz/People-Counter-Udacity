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
        print(cmd)
        print(subprocess.check_output(cmd, shell=True).decode())

