################################################################################
#
# Package   : AlphaPy
# Module    : mflow_server
# Created   : February 21, 2021
#
# uvicorn mflow_server:app --reload
#
# Copyright 2021 ScottFree Analytics LLC
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
# Imports
#

from alphapy.group import Group
from alphapy.mflow_main import get_market_config

from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
import numpy as np
import uuid
import uvicorn


#
# Initialize FastAPI
#

app = FastAPI()


#
# FastAPI Startup
#


@app.on_event("startup")
async def startup_event():
    global market_specs
    #market_specs = get_market_config()
    return


#
# Get groups
#

@app.get("/groups")
def read_groups():
    return Group.groups


@app.on_event("shutdown")
def shutdown_event():
    with open("log.txt", mode="a") as log:
        log.write("Application shutdown")


"""
@app.post("/{style}")
def get_image(style: str, file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))
    model = config.STYLES[style]
    output, resized = inference.inference(model, image)
    name = f"/storage/{str(uuid.uuid4())}.jpg"
    cv2.imwrite(name, output)
    return {"name": name}
"""


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)