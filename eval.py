from pathlib import Path
from collections import Counter
from PIL import Image
import torch
from main import model, preprocess, text_features,THRESHOLD

TEST_DIR = Path("/home/isesat/Downloads/casting_dataset/casting_data/casting_data/test")


def score(path):
    image = Image.open(path).convert("RGB")
    with torch.no_grad():
        feats = model.encode_image(preprocess(image).unsqueeze(0))
    feats /= feats.norm(dim=-1, keepdim=True)
    probs = (100.0 * feats @ text_features.T).softmax(dim=-1)
    return probs[0][1].item()


correct = 0
total = 0
verdicts = Counter()
scores = []

true_positive=0
false_positive=0
true_negative=0
false_negative=0

for label in ["ok_front", "def_front"]:
    truth = "NOK" if label == "def_front" else "OK"
    for path in sorted((TEST_DIR / label).glob("*.jpeg")):
        score_value = score(path)
        verdict = "NOK" if score_value >= THRESHOLD else "OK"
        if verdict =='NOK' and truth == 'NOK':
            true_positive+=1
        if verdict =='OK' and truth == 'NOK':
            false_negative+=1
        if verdict =='OK' and truth == 'OK':
            true_negative+=1
        if verdict =='NOK' and truth == 'OK':
            false_positive+=1
        
        correct += verdict == truth
        total += 1
        verdicts[verdict] += 1
        scores.append(score_value)

print(f"{correct}/{total} correct  =  {correct/total:.1%}")
print(verdicts)
print(f"score range: {min(scores):.3f} to {max(scores):.3f}")
recall = true_positive / (true_positive + false_negative)
precision = true_positive / (true_positive + false_positive)
print(f"recall (defects caught): {recall:.1%}")
print(f"precision (flags that were real): {precision:.1%}")