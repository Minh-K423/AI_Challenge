from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

import numpy as np
import matplotlib
matplotlib.use('QtAgg') 
import matplotlib.pyplot as plt
from PIL import Image
import pathlib

import torch
import torch.nn.functional as F

from transformers import CLIPProcessor, CLIPModel


client = QdrantClient(host="localhost", port=6333)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") 
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def process_query(query: str):
    processed_query = processor(text=query, return_tensors='pt')
    
    with torch.no_grad():
        output = model.get_text_features(
            input_ids=processed_query.input_ids,
            attention_mask=processed_query.attention_mask,
        )
        unnormed_query_vector = output.pooler_output[0].numpy()
        normalized_query_vector = unnormed_query_vector / np.linalg.norm(unnormed_query_vector)  # Normalize the query vector
        query_vector = normalized_query_vector.tolist()  # Convert to list for Qdrant  

    return query_vector 

def display_images(image_paths: list):
    plt.figure(figsize=(15, 15))
    for i, image_path in enumerate(image_paths):
        img = Image.open(image_path)
        plt.subplot(3, 2, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()


def main():
    user_query = input("Enter your search query: ")
    query_vector = process_query(user_query)
        
    search_results = client.query_points(
        collection_name="video_features_collection",
        query=query_vector,
        limit=6,
        with_payload=True,
        with_vectors=True
    ).points

    image_paths = []

    print("\nTop 6 search results:")
    for i, result in enumerate(search_results):
        video_name = result.payload.get("video_name", "Unknown")
        frame_index = result.payload.get("frame_index", "Unknown")
        title = result.payload.get("title", "No available key")
        description = result.payload.get("description", "No available key")
        score = result.score if hasattr(result, 'score') else "N/A"
        print(f"Result {i+1}: Video Name: {video_name}, Frame Index: {frame_index}, Score: {score}")

        if video_name[0:3] == "L26":
            category = int(video_name[-3:]) // 100
            image_path = pathlib.Path("/mnt/Shared_data/Data") / f"Keyframes_{video_name[0:3]}_{chr(ord('a') + category)}"  / "keyframes" / video_name / f"{frame_index:03d}.jpg"
        else: 
            image_path = pathlib.Path("/mnt/Shared_data/Data") / f"Keyframes_{video_name[0:3]}" / "keyframes" / video_name / f"{frame_index:03d}.jpg" #The / operator has higher level of priority compared to + operator so image_path = pathlib.Path("/mnt/Shared_data/Data") / "Keyframes" + video_name[0:3] / "keyframes" / video_name / f"{frame_index:03d}" + ".jpg"  would be wrong, so either change to f"..." or wrap them into (...)
    
        image_paths.append(image_path)
    
    display_images(image_paths)


if __name__ == "__main__":
    main()