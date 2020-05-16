import os
from pathlib import Path
from urllib.parse import urlparse
import typing
import validators
import requests
import tarfile
from decor import pathassert
from tqdm import tqdm
import urllib.request
import cv2
from urllib.error import HTTPError
import numpy as np


class FileUtil:

    @staticmethod
    def internet_on(url='http://www.google.com/', timeout=5):
        try:
            req = requests.get(url, timeout=timeout)
            req.raise_for_status()
            return True
        except requests.HTTPError as e:
            print("Checking internet connection failed, status code {0}.".format(
                e.response.status_code))
        except requests.ConnectionError:
            print("No internet connection available.")
        return False

    @classmethod
    def download_file(cls, file_uri: str, out_folder: str = None, force=False, unzip=True):
        try:
            assert cls.internet_on(), "Not internet connection"
            assert validators.url(file_uri), "invalid file uri parameter"
            remote_file_path = urlparse(file_uri).path
            remote_file_name = os.path.basename(remote_file_path)
            out_folder = Path(out_folder) if out_folder else Path(os.getcwd())
            out_folder.mkdir(exist_ok=True)
            out_file = out_folder.joinpath(remote_file_name)
            if not out_file.exists() or force:
                print("[INFO]: downloading file : {}".format(file_uri))
                r = requests.get(file_uri, stream=True)
                total_length = int(r.headers.get('content-length'))
                block_size = 1024  # 1 Kibibyte
                with tqdm(total=total_length, unit='iB', unit_scale=True) as t:
                    with open(str(out_file), 'wb') as f:
                        for data in r.iter_content(block_size):
                            t.update(len(data))
                            f.write(data)
                if total_length != 0 and t.n != total_length:
                    raise Exception("ERROR, something went wrong")

                print("[INFO]: File download done")
            if unzip and out_file.suffix in [".gz", ".zip"]:
                out_folder_name = remote_file_name[:remote_file_name.find('.')]
                cls.unzip_file(out_file, out_folder.joinpath(out_folder_name))
            return out_file
        except Exception as ex:
            print("[ERROR]: Error downloading the file from {}: {}".format(file_uri, ex))


    @staticmethod
    @pathassert
    def unzip_file(file_path: typing.Union[str, Path], output_folder: Path):
        output_folder.mkdir(exist_ok=True)
        ext = file_path.suffix
        if ext == ".gz":
            with tarfile.open(str(file_path)) as tar:
                dirs: [tarfile.TarInfo] = [m for m in tar.getmembers() if m.isdir()]
                files: [tarfile.TarInfo] = [m for m in tar.getmembers() if m.isfile()]
                # unzip the dirs
                for member in dirs:
                    path_parts = Path(member.name).parts
                    # ignore root folder
                    if len(path_parts) > 1:
                        folder_path = os.sep.join(Path(member.name).parts[1:])
                        folder_path = output_folder.joinpath(folder_path)
                        tar.makedir(member, folder_path)
                # unzip the files
                for member in files:
                    file_path = os.sep.join(Path(member.name).parts[1:])
                    file_path = output_folder.joinpath(file_path)
                    tar.makefile(member, file_path)
        else:
            raise IOError("Extension {} not supported yet".format(ext))

    @staticmethod
    def url2img(url):
        try:
            assert validators.url(url), "invalid url"
            resp = urllib.request.urlopen(url, timeout=30)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except HTTPError as err:
            if err.code == 404:
                raise Exception("Image not found")
            elif err.code == 403:
                raise Exception("Forbidden image, try with other one")
            else:
                raise