from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
#Load Blip
class BLIPCaptioner:
    def __init__(self, model_id="Salesforce/blip-image-captioning-base"):
        self.processor = BlipProcessor.from_pretrained(model_id, use_fast=True)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id)
        
    def generate_caption(self, image_pil, max_length=40):
        """Generates a caption for a PIL image."""
        inputs = self.processor(images=image_pil, return_tensors="pt")
        with torch.no_grad():
            ids = self.model.generate(pixel_values=inputs.pixel_values, max_length=max_length)
        caption = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return caption
