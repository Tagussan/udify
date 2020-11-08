import os
import sys
import shutil
import logging
import argparse
import tarfile
from pathlib import Path

from allennlp.common import Params
from allennlp.common.util import import_submodules
from allennlp.models.archival import archive_model

from udify import util

from flask import Flask, request, jsonify
import tempfile

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        level=logging.ERROR)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--port", default=8888, type=int, help="The port to listen")

args = parser.parse_args()

archive_path = "./udify-model.tar.gz"
if not os.path.exists(archive_path):
    print("put archive: ", archive_path)
    sys.exit(1)

import_submodules("udify")

archive_dir = Path(archive_path).resolve().parent

if not os.path.isfile(archive_dir / "weights.th"):
    with tarfile.open(archive_path) as tar:
        print("extracting model archive")
        tar.extractall(archive_dir)

config_file = archive_dir / "config.json"
print("config:", config_file)

overrides = {}
overrides["trainer"] = {"cuda_device": -1} #-1 for cpu
configs = [Params(overrides), Params.from_file(config_file)]
params = util.merge_configs(configs)

predictor = "udify_text_predictor"

print("loading model")
predictor = util.get_file_iface_predictor_with_archive(predictor, params, archive_dir)

#start server
app = Flask(__name__)
@app.route('/', methods=['POST'])
def post_json():
    json = request.get_json()
    input_text = json['text']
    temp_input = tempfile.NamedTemporaryFile()
    fp = open(temp_input.name, 'w')
    fp.write(input_text)
    fp.close()
    temp_output = tempfile.NamedTemporaryFile()
    predictor(temp_input.name, temp_output.name)
    fp = open(temp_output.name, 'r')
    result = fp.read()
    fp.close()
    temp_input.close()
    temp_output.close()
    return jsonify(result)

print("starting server")
app.run(port=args.port)
