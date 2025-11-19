"""
A ResNet101 image classification pipeline using pre-trained weights.
"""

import torch
from absl import app
from torchvision import models
from PIL import Image
from torchvision import transforms


def main(argv):
    del argv
    weights = models.ResNet101_Weights.IMAGENET1K_V1
    resnet = models.resnet101(weights=weights)
    resnet.eval()
    img = Image.open('testdata/silva.jpg')
    # It can be shown with existing PIL
    # img.show()
    # ... or with matplotlib + qt
    # plt.imshow(img)
    # plt.show()

    # Format to RESNET
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    preprocess = weights.transforms()
    batch = preprocess(img).unsqueeze(0)  # shape: [1, 3, H, W]

    with torch.no_grad():
        logits = resnet(batch)

    probs = logits.softmax(dim=1)[0]
    categories = weights.meta['categories']
    top5_prob, top5_idx = torch.topk(probs, 5)

    for p, idx in zip(top5_prob, top5_idx):
        print(f'{categories[idx]}: {p.item():.2f}')


if __name__ == '__main__':
    app.run(main)
