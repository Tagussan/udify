import os
import shutil
import logging
import argparse
import tarfile
from pathlib import Path

from allennlp.common import Params
from allennlp.common.util import import_submodules
from allennlp.models.archival import archive_model

from udify import util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        level=logging.ERROR)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("archive", type=str, help="The archive file")
parser.add_argument("input_file", type=str, help="The input file to predict")
parser.add_argument("pred_file", type=str, help="The output prediction file")

args = parser.parse_args()

import_submodules("udify")

archive_dir = Path(args.archive).resolve().parent

if not os.path.isfile(archive_dir / "weights.th"):
    with tarfile.open(args.archive) as tar:
        print("extracting model archive")
        tar.extractall(archive_dir)

config_file = archive_dir / "config.json"
print("config:", config_file)

overrides = {}
overrides["trainer"] = {"cuda_device": -1} #-1 for cpu
configs = [Params(overrides), Params.from_file(config_file)]
params = util.merge_configs(configs)

predictor = "udify_text_predictor"

util.predict_model_with_archive(predictor, params, archive_dir, args.input_file, args.pred_file)
