import numpy as np
import json
import os
import uuid

def explore_data(npy_path, json_path):
    print("="*50)
    print(f" Exploring data:... ")
    print("="*50)

    if os.path.exists(npy_path):
        features = np.load(npy_path)
        print(f"\n[+] File NPY: {npy_path}")
        print(f"    -> Data type: {features.dtype}")
        print(f"    -> Size and the matrix: {features.shape}")
    else:
        print(f"\n[-] Couldn't find {npy_path}")

    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"\n[+] File JSON: {json_path}")
        print(f"    -> Keys available: {list(metadata.keys())}")
 
        if 'description' in metadata:
            print(f"    -> Preview Description: {metadata['description'][:100]}...")
    else:
        print(f"\n[-] Couldn't find {json_path}")


def build_points_from_videos(npy_path, json_path):
    print("Hello world")

if __name__ == "__main__":
    sample_npy = "L01_V001.npy" 
    sample_json = "L01_V001.json"
    
    explore_data(sample_npy, sample_json)