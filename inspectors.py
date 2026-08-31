from abc import ABC, abstractmethod
import torch
import open_clip

class Inspector(ABC):
    name = "base"

    @abstractmethod
    def defect_score(self, image) -> float:
        """Return P(defect) between 0 and 1 for a PIL image."""

class ClipInspector(Inspector):
    name = "clip"

    def __init__(self, ok_prompt, defect_prompt):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")

        texts = tokenizer([ok_prompt, defect_prompt])
        with torch.no_grad():
            self.text_features = self.model.encode_text(texts)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def defect_score(self, image) -> float:
        with torch.no_grad():
            feats = self.model.encode_image(self.preprocess(image).unsqueeze(0))
        feats /= feats.norm(dim=-1, keepdim=True)
        probs = (100.0 * feats @ self.text_features.T).softmax(dim=-1)
        return probs[0][1].item()