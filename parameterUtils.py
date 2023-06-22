import os
import json
import yaml



def parameterLoad(filePath):
    str_ = ""
    if os.path.exists(filePath):
        file = open(filePath, mode="r")
        data = dict(yaml.safe_load(file))
    else:
        data = dict()
    if not "HSV" in data.keys():
        data["HSV"] = dict()
    if not "MayBeTarget" in data.keys():
        data["MayBeTarget"] = dict()
    return data


def parameterWrite(filePath, data):
    file = open(filePath, mode="w", newline="\n")
    data = json.dumps(data, indent=4)
    print(data)
    file.write(data)
    file.close()


# YApi QuickType插件生成，具体参考文档:https://plugins.jetbrains.com/plugin/18847-yapi-quicktype/documentation
# load parameter

from typing import List
import numpy as np


class HSV:

    def __init__(self, data: dict) -> None:
        self.upperLimit = np.array(data["upperLimit"])
        self.lowerLimit = np.array(data["lowerLimit"])


class MayBeTarget:
    area: float
    width: float
    height: float

    def __init__(self, data: dict) -> None:
        self.area = data["area"]
        self.width = data["width"]
        self.height = data["height"]


class Parameter:
    HSV: HSV
    data: dict
    kernel: int
    insideRate: float
    outsideRate: float
    MayBeTarget: MayBeTarget

    def __init__(self, jsonPath: str) -> None:
        data = parameterLoad(jsonPath)
        self.data = data
        self.HSV = HSV(data["HSV"])
        self.kernel = data["kernel"]
        self.insideRate = data["insideRate"]
        self.outsideRate = data["outsideRate"]
        self.MayBeTarget = MayBeTarget(data["MayBeTarget"])
