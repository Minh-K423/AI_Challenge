import os
import json
import numpy as np
import uuid 


def data_availability(npy_path, json_path):
    print("="*50)
    print(f" EXPLORING DATA ")
    print("="*50)

    if os.path.exists(npy_path):
        npy_existence = True
        features = np.load(npy_path)
        print(f"\n[+] NPY file: {npy_path}")
        print(f"    -> Data type: {features.dtype}")
        print(f"    -> Matrix Shape: {features.shape}")
    else:
        npy_existence = False
        print(f"\n[-] Not found: {npy_path}")

    if os.path.exists(json_path):
        json_existence = True
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"\n[+] JSON file: {json_path}")
        print(f"    -> Available fields (Keys): {list(metadata.keys())}")
        
        if 'description' in metadata:
            print(f"    -> Description Preview: {metadata['description'][:100]}...")
    else:
        json_existence = False
        print(f"\n[-] Not found: {json_path}")

    if (npy_existence and json_existence):
        return True
    else:
        return False


def generate_point(npy_path, json_path):
    video_name = os.path.splitext(os.path.basename(npy_path))[0]
    metadata = {}

    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    title = metadata.get("title", "No available key")
    description = metadata.get("description", "No available key")

    points_to_upload = []

    video_features = np.load(npy_path)
    num_frames = video_features.shape[0]

    for frame_index in range(num_frames):
        frame_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{video_name}_{frame_index}"))

        vector = video_features[frame_index].tolist() #Cause Qdrant do not support numpy array, we need to convert it to list

        payload = {
            "video_name": video_name,
            "frame_index": frame_index,
            "title": title,
            "description": description
        }

        point = {
            "id": frame_id,
            "vector": vector,
            "payload": payload
        }

        points_to_upload.append(point)

    return points_to_upload



def main():
    sample_npy = "L01_V001.npy" 
    sample_json = "L01_V001.json"
    
    if data_availability(sample_npy, sample_json):
        points = generate_point(sample_npy, sample_json)


if __name__ == "__main__":
    main()
