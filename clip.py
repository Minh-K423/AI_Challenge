import torch
import torch.nn.functional as F
from PIL import Image #Image processing library
from transformers import CLIPModel, CLIPProcessor

def top_k(results: list[dict]):
    for i in range(len(results)):
        for j in range(i,len(results)):
            print("Hello world")

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

    input = []

    for i in range(len(image)):
        input.append(processor(images = image[i], return_tensors = 'pt')) #Don't' worry, it still work:)) I have no idea why it is like that

    string = "a photo of a grey cat"
    query1 = processor(text=string, return_tensors = 'pt')

    feat = []

    with torch.no_grad():
        for i in range(len(input)):
            feat.append(model.get_image_features(pixel_values = input[i].pixel_values))

        query = model.get_text_features(input_ids=query1.input_ids, attention_mask=query1.attention_mask)

    results = []
    for i in range(len(feat)):
        results.append({"similarity": F.cosine_similarity(feat[i].pooler_output, query.pooler_output).item(),
                      "image": "Image " + str(i+1)})
        print(f"The cosine similarity between{results[i]["image"]} and the query {string} is {results[i]["similarity"]:.4f}")
    
    '''
    similarity1 = F.cosine_similarity(feat1.pooler_output, query.pooler_output)
    print(f"Độ tương đồng Cosine giữa ảnh 1 và query '{string}' là: {similarity1.item():.4f}")

    similarity2 = F.cosine_similarity(feat2.pooler_output, query.pooler_output)
    print(f"Độ tương đồng Cosine giữa ảnh 2 và query '{string}' là: {similarity3.item():.4f}")

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