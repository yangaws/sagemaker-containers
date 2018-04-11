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
import logging
import shlex
import subprocess
import traceback

import os

logging.basicConfig()
logger = logging.getLogger(__name__)

SUCCESS_CODE = 0
OUTPUT_PATH = "/opt/ml/output"


def main(args):
    # Set a global flag that indicates that we were invoked from the
    # command line interface. This is detected by Flask.run to make the
    # call into a no-op. This is necessary to avoid ugly errors when the
    # script that is loaded here also attempts to start a server.

    module_name = args[1]

    try:
        subprocess.check_call(shlex.split('python -m %s' % module_name))
        write_success()
        os._exit(SUCCESS_CODE)
    except Exception as e:
        trc = traceback.format_exc()
        message = 'Uncaught exception during training: {}\n{}\n'.format(e, trc)
        write_failure(message)

        logger.error(message)
        exit_code = e.errno if hasattr(e, 'errno') else 1
        os._exit(exit_code)


def write_success():
    write_output('success')


def write_failure(message):
    write_output('failure', message)


def write_output(file_name, message=None):
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    message = message if message else 'Training finished with {}'.format(file_name)

    with open(os.path.join(OUTPUT_PATH, file_name), 'a') as fd:
        fd.write(message)
