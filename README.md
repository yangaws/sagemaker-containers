# Sagemaker Containers

## Philosophy (inspired by requests and Flask)
Developed with a few PEP 20 idioms in mind:

- Beautiful is better than ugly.
- Explicit is better than implicit.
- Simple is better than complex.
- Complex is better than complicated.
- Readability counts.

## Guiding Principles
- Modularity:  Higher level functions are composed by lower level functions, allowing customers to implement their own version of the higher level versions.
- User Friendliness: attributes are explict and well documented. Errors messages explain the error, educating the user.
    Functions are small and well tested.

## Design

#### container_support.environment.py
contains functions to extract environment information. 

```python
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
    pass

def read_hps():
    pass


def split_hps(hps, keys=SAGEMAKER_HPS):
    return sagemaker_hps, user_hps


def read_resource_config():
    return read_config(os.path.join(INPUT_CONFIG_PATH, RESOURCE_CONFIG_FILE))


def read_input_data_config():
    return read_config(os.path.join(INPUT_CONFIG_PATH, INPUT_DATA_CONFIG_FILE))


def channel_dir(channel):
    return os.path.join(INPUT_DATA_PATH, channel)


def gpu_count():
    pass


def cpu_count():
    pass


class Environment(
    collections.namedtuple('Environment', [
        'input_dir', 'input_config_dir', 'model_dir', 'output_dir', 'hps', 'resource_config',
        'input_data_config', 'output_data_dir', 'hosts', 'channel_dirs', 'current_host', 'gpu_count',
        'cpu_count', 'module_name', 'module_dir', 'enable_metrics', 'log_level'])):

    def dict(self):
        return self._asdict()

    @classmethod
    def create(cls, session=None):
        pass
  
```


### container_support.modules
contains functions to download, validate, import, and execute modules

```python
def download(url, dst):
    pass

def install(path):
    pip.main(shlex.split('install %s' % path))

def start_metrics():
    pass

def download_and_import(url, dst, name=DEFAULT_MODULE_NAME):
    dst = download(url, tempfile.mkstemp())
    name = install(dst)
    return importlib.import_module(name)
```
### container_support.functions
contains functions to invoke user provide functions with env parameters
```python
def signature(fn):
    pass

def intersect_fn_args(fn, environment):
    args, kwargs = signature(fn)

    if kwargs:
        return environment

    return filter(lambda key, value: key in args, environment)

def call(fn, environment):
    return fn(intersect_fn_args(fn, environment))

def execute(module):
    pass
```


#### framework-example.py
This example shows how to write a framework
```python
import sagemaker_containers as smc

def keras_framework_training_fn():
    # creates the environment
    env = smc.Environment.create()

    # downloads and import the user module
    mod = smc.modules.download_and_import(env.module_dir, env.module_name)

    # call train function with matching environment parameters
    model = smc.functions.call(mod.train, environment=env)
    if model:
        if hasattr(mod, 'save'):
            mod.save(model, env.model_dir)
        else:
            model_file = os.path.join(env.model_dir, 'saved_model')
            model.save(model_file)


if __name__ == '__main__':
    keras_framework_training_fn()
```

#### framework-example-adding-attributes-to-env.py

```python

# this class extends the environment
class PYTorchEnvironment(smc.Environment):
    def __init__(self):

    def create(cls, session=None):
        self.world_size = get_world_size(len(self.hosts), self.gpu_count))

def get_world_size(hosts_count, cpu_count):
    return hosts_count * cpu_count

def py_torch_framework_training_fn():
    env = PYTorchEnvironment.create()

    mod = smc.modules.download_and_import(env.module_dir, env.module_name)

    model = smc.functions.call_with_env(mod.train, environment=env)

    # ...


if __name__ == '__main__':
    py_torch_framework_training_fn()
```

### Dockerfile
ML frameworks can opt in to use the module runner or not. The module runner executes the module and reports success/failure
```
ENTRYPOINT ["python", "-m", "sagemaker_containers.runner", "keras_container.start"]
```
or
```
ENTRYPOINT ["sagemaker_containers_runner", "keras_container.start"]
```

### runner
Inspired by Flask, pytest, gunicorn
```python
def main(args):
    module = args[1]

    env = smc.Environment.create()

    if env.script_mode:
        module = smc.modules.download_and_import(env.module_dir, env.module_name)

    try:
        smc.modules.execute(module)

        write_output(env.output_dir, 'success')
        os._exit(SUCCESS_CODE)
    except Exception as e:
        os._exit(exit_code)
```

### customer_script.py
```python
import os
import keras
import numpy as np

def train(training_dir, hyperparameters):
    data = np.load(os.path.join(training_dir, hyperparameters['training_data_file']))
    x_train, y_train = data['features'], keras.utils.to_categorical(data['labels'])

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(64, activation='relu', input_dim=20))
    model.add(keras.layers.Dropout(0.5))

    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=20, batch_size=128)
    return model
```

### customer_script.py - script mode
```python
import argparse
import os

import sagemaker_containers as smc
import numpy as np
import tensorflow as tf

env = smc.Environment.create() 

parser = argparse.ArgumentParser()
parser.add_argument('--training-data-dir', type=str, default=env.channel_dirs['training'])
parser.add_argument('--batch-size', type=int, default=env.batch_size.to_int())
parser.add_argument('--model-dir', type=str, default=env.model_dir)
parser.add_argument('--lr', type=float, default=env['learning-rate'].to_float())

args = parser.parse_args()

data = np.load(os.path.join(args.training_data_dir, 'training_data.npz'))
x_train, y_train = data['features'], keras.utils.to_categorical(data['labels'])

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_dim=20))
model.add(tf.keras.layers.Dropout(0.5))

sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20, batch_size=args.batch_size)

# saves the model in the end of training
model.save(os.path.join(args.model_dir, 'saved_model.h5'))
```

### for the future - script mode namespacing

```python
import container_support as smc

# creates PYTorchEnvironment
env = smc.pytorch.Environment.create()

# tensorflow specific dataset
dataset = smc.tensorflow.PipeModeDataset()

# invoking chainer special function broadcast_to
smc.chainer.broadcast_to(x, shape)
```

## License

This library is licensed under the Apache 2.0 License. 
