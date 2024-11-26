# Vision Transformers (ViT) Implementation

This repository contains an implementation of a **Vision Transformer (ViT)**, inspired by the original paper ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929). The model is implemented using PyTorch and applied to the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset for image classification.

---

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Inference Example](#inference-example)

---

## Features
- Custom implementation of:
  - **Patch Embedding**
  - **Transformer Encoder**
  - **Vision Transformer (ViT)**
- Configurable hyperparameters for model architecture, optimizer, and training process.
- Preprocessing pipeline for the CIFAR-10 dataset.
- Early stopping to prevent overfitting.

---

## Requirements
- Python 3.8+
- PyTorch 2.0+
- torchvision
- tqdm
- numpy
- matplotlib
- einops (for the first version of Patch Embeddings)

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## Usage

Open the project's directory with all necessary classes and functions
```bash
git clone https://github.com/RelentlessViper/vision-transformers.git

cd vision-transformers/code/vit_tools
```

### Training the Model
Download [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) Dataset:
```Python
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader

dataset_pars = {
    'image_size': 32,
    'batch_size': 80
}

train_transforms = T.Compose(
    [
        T.Resize((dataset_pars['image_size'])),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
)

test_transforms = T.Compose(
    [
        T.Resize((dataset_pars['image_size'])),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
)

train_data = datasets.CIFAR10(root='cifar10/train', train=True, download=True, transform=train_transforms)
test_data = datasets.CIFAR10(root='cifar10/test', train=False, download=True, transform=test_transforms)

train_loader = DataLoader(train_data, batch_size=dataset_pars['batch_size'], shuffle=True)
test_loader = DataLoader(test_data, batch_size=['batch_size'], shuffle=False)
```

Initialize model, dataloaders with all hyperparameters set:
```Python
from vit import VIT
from train import train_model

model_pars = {
    'num_layers': 12,
    'latent_size': 768,
    'num_heads': 12,
    'num_classes': 10,
    'batch_size': dataset_pars['batch_size'],
    'dropout': 0.5,
    'patch_embedding_version': 'v2'
}
model = VIT(
    **model_pars,
    patch_embedding_version='v2'
).to(device)


optimizer_pars = {
    'lr': 0.0001,
    'weight_decay': 0.0001
}
optimizer = optim.Adam(model.parameters(), **optimizer_pars)
criterion = nn.CrossEntropyLoss()
```

Initialize one training iteration:
```Python
training_pars = {
    'num_epochs': 40,
    'early_stopping_patience': 15
}

training_loop_counter = 0

train_model(
    model,
    optimizer,
    criterion,
    train_loader,
    test_loader,
    training_loop_counter,
    device=device,
    **training_pars
)
training_loop_counter += 1
```

### Inference Example

```Python
from PIL import Image

model.eval()

image = Image.open('path_to_image.jpg')
image = test_transforms(image).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(image.to(device)) # Assuming that batch size = 1
    prediction = torch.argmax(logits, dim=1).item()
    print(f'Predicted class: {prediction}')
```
