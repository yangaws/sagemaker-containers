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

import importlib
import inspect
import logging
import shlex
import subprocess
import tarfile
import tempfile

import boto3
import pip
import six
from six.moves.urllib.parse import urlparse

logging.basicConfig()
logger = logging.getLogger(__name__)

SAGEMAKER_PROGRAM_PARAM = "sagemaker_program"
USER_SCRIPT_ARCHIVE_PARAM = "sagemaker_submit_directory"
CLOUDWATCH_METRICS_PARAM = "sagemaker_enable_cloudwatch_metrics"
CONTAINER_LOG_LEVEL_PARAM = "sagemaker_container_log_level"
JOB_NAME_PARAM = "sagemaker_job_name"
DEFAULT_MODULE_NAME = "default_user_module_name"


def download_s3_file(url, dst):
    bucket, key = parse_s3_url(url)

    s3 = boto3.resource('s3')
    s3.Bucket(bucket).download_file(key, dst)
    return dst


def parse_s3_url(url):
    """ Returns an (s3 bucket, key name/prefix) tuple from a url with an s3 scheme
    """
    url = urlparse(url)

    if url.scheme != "s3":
        raise ValueError("Expecting 's3' scheme, got: %s in %s" % (url.scheme, url))
    return url.netloc, url.path.lstrip('/')


def untar(src, dst):
    with open(src, 'rb') as f:
        with tarfile.open(mode='r:gz', fileobj=f) as t:
            t.extractall(path=dst)
            return dst


def create_python_package(path):
    pass


def pip_install(package):
    pip.main(shlex.split('install %s' % package))


def start_metrics():
    logger.info("starting metrics service")
    subprocess.Popen(['telegraf', '--config', '/usr/local/etc/telegraf.conf'])


def import_s3_module(url, dst, name=DEFAULT_MODULE_NAME):
    path = download_s3_file(url, tempfile.mkstemp())
    dst = untar(path, dst)

    create_python_package(dst)
    pip_install(dst)

    return importlib.import_module(name)


def signature(fn):
    if six.PY2:
        arg_spec = inspect.getargspec(fn)
        return arg_spec.args, arg_spec.keywords

    sig = inspect.signature(fn)
    args = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]

    kwargs = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_KEYWORD
    ]
    kwargs = kwargs[0] if kwargs else None
    return args, kwargs


def intersect_fn_args(fn, environment):
    args, kwargs = signature(fn)

    if kwargs:
        return environment

    return filter(lambda key, value: key in args, environment)


def call_fn(fn, environment):
    return fn(intersect_fn_args(fn, environment))
