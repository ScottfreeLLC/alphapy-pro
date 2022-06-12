################################################################################
#
# Package   : AlphaPy
# Module    : mflow_server
# Created   : February 21, 2021
#
# Copyright 2022 ScottFree Analytics LLC
# Mark Conway & Robert D. Scott II
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################


#
# HOW TO RUN:
#
# export ALPHAPY_ROOT=/Users/markconway/Projects/alphapy-data
# uvicorn mflow_server:app --reload
#


#
# Imports
#

from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
import logging
import os
from pathlib import Path
import sys
import uuid
import uvicorn

from alphapy.group import Group
from alphapy.mflow_main import get_market_config
from alphapy.model import get_model_config


#
# Initialize FastAPI
#

app = FastAPI()


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# FastAPI Startup
#

@app.on_event("startup")
async def startup_event():
    # Initialize Logging
    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                        filename="mflow_server.log", filemode='a', level=logging.INFO,
                        datefmt='%m/%d/%y %H:%M:%S')
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                  datefmt='%m/%d/%y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    # Start the pipeline
    logger.info('*'*80)
    logger.info("Market Flow Server Start")
    logger.info('*'*80)
    # Get the AlphaPy environment variable
    alphapy_root = os.environ.get('ALPHAPY_ROOT')
    if not alphapy_root:
        root_error_string = "ALPHAPY_ROOT environment variable must be set"
        logger.info(root_error_string)
        sys.exit(root_error_string)
    # Finish Startup
    return


#
# Get groups
#

@app.get("/groups")
def request_groups():
    return Group.groups


#
# Get market specifications
#

@app.get("/market_config")
def request_market_config(project_root):
    cfg, specs = get_market_config(project_root)
    return cfg, specs


#
# Get model specifications
#

@app.get("/model_config")
def request_model_config(project_root):
    cfg, specs = get_model_config(project_root)
    return cfg, specs


#
# Get paths
#

@app.get("/paths")
def request_paths(alphapy_specs):
    root_directory = alphapy_specs['mflow']['project_root']
    paths = []
    for path in Path(root_directory).rglob('market.yml'):
        paths.append(path)
    return paths


#
# Get projects
#

@app.get("/projects")
def request_projects(alphapy_specs):
    root_directory = alphapy_specs['mflow']['project_root']
    projects = []
    for path in Path(root_directory).rglob('market.yml'):
        path_str = str(path).split('/')
        projects.append(path_str[-3])
    return projects


#
# Shut down the application
#

@app.on_event("shutdown")
def shutdown_event():
    # Stop the pipeline
    logger.info('*'*80)
    logger.info("Market Flow Server End")
    logger.info('*'*80)


#
# Main Program
#

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)