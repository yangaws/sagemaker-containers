# Sagemaker Containers

## Guiding Principles
- First Class Functions: the entire API is a composition of functions that can be individually used.
- Modularity:  Higher level functions are composed by lower level functions, allowing customers to implement their own version of the higher level versions.
- User Friendliness: attributes are explict and well documented. Errors messages explain the error, educating the user.

## Design

#### environment.py
Environment information including configuration data, environment variables, hyperparameters.

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


def load_config(path):

def load_hyperparameters():

def load_resource_config():

def load_input_data_config():

def get_channel_dir(channel):

def get_available_gpus():

def get_available_cpus():

def create_environment():
    resource_config = load_resource_config()
    current_host = resource_config['current_host']
    hosts = resource_config['hosts']

    input_data_config = load_input_data_config()
    channel_dirs = {channel: get_channel_dir(channel) for channel in input_data_config}

    available_cpus = get_available_cpus()
    available_gpus = get_available_gpus()

    env = TrainerEnvironment(input_dir=INPUT_PATH,
                             input_config_dir=INPUT_CONFIG_PATH,
                             model_dir=MODEL_PATH,
                             output_dir=OUTPUT_PATH,
                             output_data_dir=OUTPUT_DATA_PATH,
                             current_host=current_host,
                             hosts=hosts,
                             channel_dirs=channel_dirs,
                             available_gpus=available_gpus,
                             available_cpus=available_cpus,
                             hyperparameters=load_hyperparameters(),
                             resource_config=resource_config,
                             input_data_config=load_input_data_config())
    return env

Environment = collections.namedtuple('Environment', [
                          'input_dir', 'input_config_dir', 'model_dir', 'output_dir', 'hyperparameters', 
                          'resource_config', 'input_data_config', 'output_data_dir', 'hosts', 'channel_dirs', 
                          'current_host', 'available_gpus', 'available_cpus'])):
```

#### trainer.py
Training utilities.

```python
# default sagemaker hyperparameters
SAGEMAKER_PROGRAM_PARAM = "sagemaker_program"
USER_SCRIPT_ARCHIVE_PARAM = "sagemaker_submit_directory"
CLOUDWATCH_METRICS_PARAM = "sagemaker_enable_cloudwatch_metrics"
CONTAINER_LOG_LEVEL_PARAM = "sagemaker_container_log_level"
JOB_NAME_PARAM = "sagemaker_job_name"
DEFAULT_USER_MODULE_NAME = "default_user_module_name"

def download_file_from_s3(uri, destination) 

def extract_tar_file(source, target)

def create_python_package(file_name)

def install_python_package(file_name)

def import_module(module_name)

def start_metrics()

def import_module_from_s3(uri, module_name=DEFAULT_USER_MODULE_NAME)
  tmp = tempfile.temp
  download_file_from_s3(uri, tmp) 
  
  tmp2 = tempfile.temp
  extract_tar_file(tmp, tmp2)

  create_python_package(file_name)
  install_python_package(file_name)
  
  return import_module(module_name)

def fn_signature(fn)

def intersect(dict, dict)

def intersect_fn_parameters(fn, hyperparameters)
  dict = fn_signature(fn)
  return intersect(dict, hyperparameters)
  
def invoke_fn(fn, dict)
  return fn(intersect_fn_parameters(fn, dict))

def training_from_fn(uri, hyperparameters, module_name=DEFAULT_USER_MODULE_NAME)
  mod = import_module_from_s3(uri, module_name)
  
  return invoke_fn(mod.train, hyperparameters)
  
```
#### framework-example.py

```python
def keras_framework_training_fn():
    env = create_environment()
    
    model = training_from_fn(env.user_script_name, env.hyperparameters)
    ...

```
## License

This library is licensed under the Apache 2.0 License. 
