# src/create_demo_dataset.py
# Crée un dataset de démo pour tester le pipeline

import torchvision
import torchvision.transforms as transforms
import os
import shutil
from PIL import Image
import numpy as np

def create_demo_dataset():
    """
    Télécharge CIFAR-10 et le réorganise
    dans la structure attendue par notre pipeline
    """
    
    print("📥 Téléchargement CIFAR-10...")
    
    # Télécharger CIFAR-10
    dataset = torchvision.datasets.CIFAR10(
        root='data/raw',
        train=True,
        download=True
    )
    
    # Noms des classes
    classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Créer structure train/val/test
    splits = {
        'train' : (0,    4000),   # 4000 images
        'val'   : (4000, 4500),   # 500 images
        'test'  : (4500, 5000),   # 500 images
    }
    
    for split, (start, end) in splits.items():
        for cls in classes:
            os.makedirs(
                f'data/{split}/{cls}', 
                exist_ok=True
            )
    
    print("📂 Organisation des images...")
    
    for split, (start, end) in splits.items():
        count = {cls: 0 for cls in classes}
        
        for idx in range(len(dataset)):
            image, label = dataset[idx]
            cls_name     = classes[label]
            
            # Compter par classe pour équilibrer
            if count[cls_name] < (end - start) // len(classes):
                # Sauvegarder image
                img_path = (
                    f'data/{split}/{cls_name}/'
                    f'{cls_name}_{idx:05d}.jpg'
                )
                image.save(img_path)
                count[cls_name] += 1
        
        total = sum(count.values())
        print(f"   ✅ {split}: {total} images")
    
    print("\n✅ Dataset créé !")
    print("Structure :")
    print("data/train/airplane/  → images")
    print("data/train/car/       → images")
    print("... etc")

if __name__ == "__main__":
    create_demo_dataset()
