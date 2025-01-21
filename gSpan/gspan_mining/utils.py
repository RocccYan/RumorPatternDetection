import json
import os



def read_json(json_file):
    with open(json_file, 'r') as fh:
        json_data = json.load(fh)
    return json_data

def save_as_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, ensure_ascii=False, indent=4)

def mkdir(dir_path):
    if os.path.exists(dir_path):
        return 
    else:
        os.mkdir(dir_path)