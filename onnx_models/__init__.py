# from .resnet.resnet50 import *
# from .yolo import *
import os


black_list = ["__pycache__", "__init__.py"]

def traversal(path):
    for p in os.listdir(path):
        if p in black_list:
            continue
        leaf_path = os.path.join(path, p)
        if os.path.isdir(leaf_path):
            traversal(leaf_path)
        else:
            if leaf_path.endswith(".py"):
                leaf_path = leaf_path[:-3]
                module = leaf_path.replace("/", ".")
                # print("import module: ", module)
                __import__(module)

traversal("onnx_models")
