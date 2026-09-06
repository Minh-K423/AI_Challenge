import heapq

#Library for AI and handling
import torch
import torch.nn.functional as F
from PIL import Image #Image processing library
from transformers import CLIPModel, CLIPProcessor

def main():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") 
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    path = ["Image/images1.jpg","Image/images2.jpg","Image/images3.jpg"]
    image = []
    
    for i in range(len(path)):
        image.append(Image.open(path[i]))

    input_results = processor(images = image, return_tensors = 'pt')

    string = "a photo of a grey cat"
    query = processor(text=string, return_tensors = 'pt')

    feat = []

    with torch.no_grad():
        feat = model.get_image_features(pixel_values = input_results.pixel_values)
        query_vector = model.get_text_features(input_ids=query.input_ids, attention_mask=query.attention_mask)

    results = []

    print(query_vector)

    '''for i in range(len(feat.pooler_output)):
        results.append({"similarity": F.cosine_similarity(feat.pooler_output[i:i+1], query_vector.pooler_output).item(),
                      "image": "Image " + str(i+1)})
    
    top_k_results = heapq.nlargest(5, results, key = lambda item: item["similarity"])
    for i in range(5):
        try:
            print(f"Top {i+1}: {top_k_results[i]["image"]} with the similarity of {top_k_results[i]["similarity"]}")
        except IndexError: 
            print(f"Top {i+1}: None")'''

if __name__ == "__main__":
    main()