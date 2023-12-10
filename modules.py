import random
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor


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
