import random
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import os

@st.cache_resource
def load_model():
  
  repo = "joseluhf11/sign_language_classification_v1"
  
  image_processor = AutoImageProcessor.from_pretrained(repo)
  model = AutoModelForImageClassification.from_pretrained(repo)

return image_processor, model



def image2text(images):
  texto = ""

  for image in images:
    
    image_processor, model = load_model()
  
    encoding = image_processor(image.convert("RGB"), return_tensors="pt")
  
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
  
    predicted_class_idx = logits.argmax(-1).item()
  
    texto = texto + model.config.id2label[predicted_class_idx]
    
  return texto


def text2image(text):
    # Directory where the images for letters and space are located
    images_directory = "Lenguaje signos"

    # Create a list to store the paths of resulting images
    resulting_images = []

    # Iterate through the text
    for character in text:
        # If the character is a space, assign an image for space
        if character == ' ':
            image_path = os.path.join(images_directory, '_.jpg')
        else:
            # If it's a letter, assume that the images have names like 'A.jpg', 'B.jpg', etc.
            image_path = os.path.join(images_directory, f'{character.upper()}.jpg')

        # Add the image path to the list
        resulting_images.append(image_path)

    return resulting_images


