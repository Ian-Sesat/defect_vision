import open_clip
import torch
from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

image = preprocess(Image.open("part.jpg")).unsqueeze(0)
texts = tokenizer([
    "a photo of a clean metal surface",
    "a photo of a scratched metal surface",
])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(texts)

print("image:", image_features.shape)
print("text: ", text_features.shape)