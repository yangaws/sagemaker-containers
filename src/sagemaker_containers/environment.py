# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

from collections import defaultdict

import boto3
import yaml
import logging
import multiprocessing

import os
import shlex
import subprocess

import collections

logging.basicConfig()
logger = logging.getLogger(__name__)

BASE_PATH = "/opt/ml"
MODEL_PATH = "/opt/ml/model"
INPUT_PATH = "/opt/ml/input"
INPUT_DATA_PATH = "/opt/ml/input/data"
OUTPUT_PATH = "/opt/ml/output"
INPUT_CONFIG_PATH = "/opt/ml/input/config"
OUTPUT_DATA_PATH = "/opt/ml/output/data"

HYPERPARAMETERS_FILE = "hyperparameters.json"
RESOURCE_CONFIG_FILE = "resourceconfig.json"
INPUT_DATA_CONFIG_FILE = "inputdataconfig.json"

PROGRAM_PARAM = "sagemaker_program"
SUBMIT_DIR_PARAM = "sagemaker_submit_directory"
ENABLE_METRICS_PARAM = "sagemaker_enable_cloudwatch_metrics"
LOG_LEVEL_PARAM = "sagemaker_container_log_level"
JOB_NAME_PARAM = "sagemaker_job_name"
DEFAULT_MODULE_NAME_PARAM = "default_user_module_name"
REGION_PARAM_NAME = 'sagemaker_region'

SAGEMAKER_HPS = [PROGRAM_PARAM, SUBMIT_DIR_PARAM, ENABLE_METRICS_PARAM,
                 LOG_LEVEL_PARAM, JOB_NAME_PARAM, DEFAULT_MODULE_NAME_PARAM]

CURRENT_HOST_ENV = "CURRENT_HOST"
JOB_NAME_ENV = "JOB_NAME"
USE_NGINX_ENV = "SAGEMAKER_USE_NGINX"


def read_config(path):
    with open(path, 'r') as f:
        return yaml.load(f)


class Caster(str):
    def to_int(self):
        return int(self)

    def to_float(self):
        return float(self)

    def to_list(self):
        return list(self)


def read_hps():
    seq = read_config(os.path.join(INPUT_CONFIG_PATH, HYPERPARAMETERS_FILE))
    return defaultdict(Caster, seq=seq)


def split_hps(hps, keys=SAGEMAKER_HPS):
    not_keys = hps.keys - keys

    return {k: v for k, v in hps.items() if k in keys}, {k: v for k, v in hps.items() if k in not_keys}


def read_resource_config():
    return read_config(os.path.join(INPUT_CONFIG_PATH, RESOURCE_CONFIG_FILE))


def read_input_data_config():
    return read_config(os.path.join(INPUT_CONFIG_PATH, INPUT_DATA_CONFIG_FILE))


def channel_dir(channel):
    """ Returns the directory containing the channel data file(s) which is:
    - <self.base_dir>/input/data/<channel>
    Returns:
        (str) The input data directory for the specified channel.
    """
    return os.path.join(INPUT_DATA_PATH, channel)


def gpu_count():
    """The number of gpus available in the current container.

    Returns:
        (int): number of gpus available in the current container.
    """
    try:
        cmd = shlex.split('nvidia-smi --list-gpus')
        output = str(subprocess.check_output(cmd))
        return sum([1 for x in output.split('\n') if x.startswith('GPU ')])
    except OSError:
        logger.warning("No GPUs detected (normal if no gpus installed)")
        return 0


def cpu_count():
    return multiprocessing.cpu_count()


def create_environment(session=None):
    """
    Returns: an instance of `TrainerEnvironment`
    """
    session = session if session else boto3.Session()

    resource_config = read_resource_config()
    current_host = resource_config['current_host']
    hosts = resource_config['hosts']

    input_data_config = read_input_data_config()
    channel_dirs = {channel: channel_dir(channel) for channel in input_data_config}

    hps, sagemaker_hps = split_hps(read_hps())

    sagemaker_region = sagemaker_hps.get(REGION_PARAM_NAME, session.region_name)

    os.environ[JOB_NAME_ENV] = sagemaker_hps[JOB_NAME_PARAM]
    os.environ[CURRENT_HOST_ENV] = current_host
    os.environ[REGION_PARAM_NAME.upper()] = sagemaker_region

    env = Environment(input_dir=INPUT_PATH,
                      input_config_dir=INPUT_CONFIG_PATH,
                      model_dir=MODEL_PATH,
                      output_dir=OUTPUT_PATH,
                      output_data_dir=OUTPUT_DATA_PATH,
                      current_host=current_host,
                      hosts=hosts,
                      channel_dirs=channel_dirs,
                      gpu_count=gpu_count(),
                      cpu_count=cpu_count(),
                      hps=hps,
                      resource_config=resource_config,
                      input_data_config=read_input_data_config(),
                      module_name=sagemaker_hps.get(PROGRAM_PARAM),
                      module_dir=sagemaker_hps.get(SUBMIT_DIR_PARAM),
                      enable_metrics=hps.get(ENABLE_METRICS_PARAM, False),
                      log_level=hps.get(LOG_LEVEL_PARAM, logging.INFO)
                      )
    return env


class Environment(
    collections.namedtuple('Environment', [
        'input_dir', 'input_config_dir', 'model_dir', 'output_dir', 'hps', 'resource_config',
        'input_data_config', 'output_data_dir', 'hosts', 'channel_dirs', 'current_host', 'gpu_count',
        'cpu_count', 'module_name', 'module_dir', 'enable_metrics', 'log_level'])):
    """Provides access to aspects of the training environment relevant to training jobs, including
    hyperparameters, system characteristics, filesystem locations, environment variables and configuration settings.

    Example on how a script can use training environment:
        ```
        import os
        import numpy as np


        from trainer.environment import create_training_environment
        env = create_training_environment()

        from keras.applications.resnet50 import ResNet50

        # get the path of the channel 'training' from the inputdataconfig.json file
        training_dir = env.channel_dirs['training']

        # get a the hyperparameter 'training_data_file' from hyperparameters.json file
        file_name = hyperparameters['training_data_file']

        # get the folder where the model should be saved
        model_dir = env.model_dir

        data = np.load(os.path.join(training_dir, training_data_file))

         x_train, y_train = data['features'], keras.utils.to_categorical(data['labels'])

        model = ResNet50(weights='imagenet')

        # unfreeze the model to allow fine tuning
        ...

        model.fit(x_train, y_train)

        # save the model in the end of training
        model.save(os.path.join(model_dir, 'saved_model'))
        ```
    """

    def __new__(cls, input_dir, input_config_dir, model_dir, output_dir, hps, resource_config,
                input_data_config, output_data_dir, hosts, channel_dirs, current_host, gpu_count,
                cpu_count, module_name, module_dir, enable_metrics, log_level):
        """

        Args:
            input_dir: The input_dir, e.g. /opt/ml/input/, is the directory where SageMaker saves input data
                        and configuration files before and during training. The input data directory has the
                        following subdirectories: config (`input_config_dir`) and data (`input_data_dir`)

            input_config_dir: The directory where standard SageMaker configuration files are located,
                        e.g. /opt/ml/input/config/.

                        SageMaker training creates the following files in this folder when training starts:
                            - `hyperparameters.json`: Amazon SageMaker makes the hyperparameters in a CreateTrainingJob
                                    request available in this file.
                            - `inputdataconfig.json`: You specify data channel information in the InputDataConfig
                                    parameter in a CreateTrainingJob request. Amazon SageMaker makes this information
                                    available in this file.
                            - `resourceconfig.json`: name of the current host and all host containers in the training

                        More information about these files can be find here:
                            https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html

            model_dir:

            output_dir: The directory where training success/failure indications will be written, e.g. /opt/ml/output.
                        To save non-model artifacts check `output_data_dir`.

            hyperparameters: An instance of `HyperParameters` containing the training job hyperparameters.

            resource_config: A dict<string, string> with the contents from /opt/ml/input/config/resourceconfig.json.
                            It has the following keys:
                                - current_host: The name of the current container on the container network.
                                    For example, 'algo-1'.
                                -  hosts: The list of names of all containers on the container network,
                                    sorted lexicographically. For example, `["algo-1", "algo-2", "algo-3"]`
                                    for a three-node cluster.

            input_data_config: A dict<string, string> with the contents from /opt/ml/input/config/inputdataconfig.json.

                                For example, suppose that you specify three data channels (train, evaluation, and
                                validation) in your request. This dictionary will contain:

                                {"train": {
                                    "ContentType":  "trainingContentType",
                                    "TrainingInputMode": "File",
                                    "S3DistributionType": "FullyReplicated",
                                    "RecordWrapperType": "None"
                                },
                                "evaluation" : {
                                    "ContentType": "evalContentType",
                                    "TrainingInputMode": "File",
                                    "S3DistributionType": "FullyReplicated",
                                    "RecordWrapperType": "None"
                                },
                                "validation": {
                                    "TrainingInputMode": "File",
                                    "S3DistributionType": "FullyReplicated",
                                    "RecordWrapperType": "None"
                                }}

                                You can find more information about /opt/ml/input/config/inputdataconfig.json here:
                                https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html#your-algorithms-training-algo-running-container-inputdataconfig

            output_data_dir: The dir to write non-model training artifacts (e.g. evaluation results) which will be
                        retained by SageMaker, e.g. /opt/ml/output/data.

                        As your algorithm runs in a container, it generates output including the status of the
                        training job and model and output artifacts. Your algorithm should write this information
                        to the this directory.

            hosts: The list of names of all containers on the container network, sorted lexicographically.
                    For example, `["algo-1", "algo-2", "algo-3"]` for a three-node cluster.

            channel_dirs:   A dict[string, string] containing the data channels and the directories where the training
                            data was saved.

                            When you run training, you can partition your training data into different logical
                            "channels". Depending on your problem, some common channel ideas are: "train", "test",
                             "evaluation" or "images',"labels".

                            The format of channel_input_dir is as follows:

                                - `channel`[key] - the name of the channel defined in the input_data_config.
                                - `training data path`[value] - the path to the directory where the training data is
                                saved.

            current_host: The name of the current container on the container network. For example, 'algo-1'.

            gpu_count: The number of gpus available in the current container.

            cpu_count: The number of cpus available in the current container.

        Returns:
            A `TrainerEnvironment` object.
        """
        return super(Environment, cls).__new__(cls,
                                               input_dir, input_config_dir, model_dir, output_dir,
                                               hps, resource_config, input_data_config,
                                               output_data_dir, hosts, channel_dirs, current_host,
                                               gpu_count, cpu_count, module_name, module_dir,
                                               enable_metrics, log_level)
