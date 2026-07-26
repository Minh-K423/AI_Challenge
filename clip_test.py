import heapq

#Library for AI and handling
import torch
import torch.nn.functional as F
from PIL import Image #Image processing library
from transformers import CLIPModel, CLIPProcessor

def partition(array: list[dict], low, high):
  pivot = array[high]["similarity"]
  i = low - 1

  for j in range(low, high):
     if array[j]["similarity"] >= pivot:
       i += 1
       array[i], array[j] = array[j], array[i] #Swap but actually "Pythonic"

  array[i+1], array[high] = array[high], array[i+1]
  return i+1

def quicksort(array: list[dict], low=0, high=None):
  if high is None:
    high = len(array) - 1

  if low < high:
    pivot_index = partition(array, low, high)
    quicksort(array, low, pivot_index-1)
    quicksort(array, pivot_index+1, high)

def top_k(results: list[dict], k: int):

    quicksort(results)

    print(f"Top {k} result is: ")
    for i in range(k):
        if i > (len(results) - 1):
           print(f"Top {i+1}: None ")
        else: 
           print(f"Top {i+1}: {results[i]['image']} with the similarity of {results[i]['similarity']} ")

def chunking(images_batch: list, chunk_size = 32) -> list[list]:
    chunks = []
    start = 0
    end = len(images_batch)

    while start < end:
        end_of_chunk = start + chunk_size
        chunk = images_batch[start:end_of_chunk]
        start = start + chunk_size
        chunks.append(chunk)

    return chunks

def one_line_chunking(images_batch: list, chunk_size = 32):
    return [images_batch[i:i+chunk_size] for i in range(0, len(images_batch), chunk_size)]

def advanced_chunking(images_batch: list, chunk_size = 32):
    for i in range(0, len(images_batch), chunk_size):
        yield images_batch[i:i+chunk_size]

def main():
    # You import the model ,great but it is just how the model learn. 
    # 
    # Now you definitely don't want to train it from scratch, so you use the pretrained version of a certain similar model using 'from_pretrained' method with the parameter being the url to that model

    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") 
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    path = ["Image/images1.jpg","Image/images2.jpg","Image/images3.jpg"]
    image = []
    
    for i in range(len(path)):
        image.append(Image.open(path[i]))

    input_results = processor(images = image, return_tensors = 'pt')
    
    string = "a photo of a grey cat"
    query1 = processor(text=string, return_tensors = 'pt')

    feat = []

    with torch.no_grad():
        
        feat = model.get_image_features(pixel_values = input_results.pixel_values)

        query = model.get_text_features(input_ids=query1.input_ids, attention_mask=query1.attention_mask)

    print(feat)

    results = []
    for i in range(len(feat.pooler_output)):
        results.append({"similarity": F.cosine_similarity(feat.pooler_output[i:i+1], query.pooler_output).item(),
                      "image": "Image " + str(i+1)})
        # print(f"The cosine similarity between{results[i]["image"]} and the query {string} is {results[i]["similarity"]:.4f}")
    
    top_k_results = heapq.nlargest(5, results, key = lambda item: item["similarity"])
    for i in range(5):
        try:
            print(f"Top {i+1}: {top_k_results[i]["image"]} with the similarity of {top_k_results[i]["similarity"]}")
        except IndexError: 
            print(f"Top {i+1}: None")

    '''
    similarity1 = F.cosine_similarity(feat1.pooler_output, query.pooler_output)
    print(f"Độ tương đồng Cosine giữa ảnh 1 và query '{string}' là: {similarity1.item():.4f}")

    similarity2 = F.cosine_similarity(feat2.pooler_output, query.pooler_output)
    print(f"Độ tương đồng Cosine giữa ảnh 2 và query '{string}' là:{similarity3.item():.4f}")

    similarity3 = F.cosine_similarity(feat3.pooler_output, query.pooler_output)
    print(f"Độ tương đồng Cosine giữa ảnh 3 và query '{string}' là: {similarity2.item():.4f}")
    '''

    # print(query1)
    # print(feat1)#
    # print(type(feat1))
    # dir(feat1)

    # print(feat1.pooler_output.shape) 


if __name__ == "__main__":
    main()