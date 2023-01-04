import cv2
import numpy as np
import pickle
import json
import os   
import glob
def get_all_data():
    try:
        with open("/opt/ml/input/data/output_pkl/output.pkl", 'rb') as f: 
            return pickle.load(f)
    except FileNotFoundError:
        return {}


def get_all_name():
    try:
        with open("/opt/ml/config.json", 'r') as f: 
            return json.load(f)
    except FileNotFoundError:
        return {}


pkl = get_all_data()
pkl = np.array(pkl)
file_name = get_all_name()['test_path']
path = os.path.join(file_name,"*")
file_path = glob.glob(path) 
file_path = sorted(file_path)

for mask, path in zip(pkl,file_path):
    cv2.imwrite(img=mask,filename="/opt/ml/input/data/test/mask/"+path[29:33]+".png")