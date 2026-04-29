# src/dataset.py
# Charge et prépare les images pour l'entraînement

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml

# ============================================
# CHARGER LA CONFIG
# ============================================
def load_config(path='configs/config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

# ============================================
# TRANSFORMATIONS / AUGMENTATIONS
# ============================================
def get_transforms(mode='train', image_size=224):
    """
    mode='train' → augmentations pour généraliser
    mode='val'   → juste redimensionner + normaliser
    """
    
    if mode == 'train':
        return A.Compose([
            # Redimensionner toutes les images à la même taille
            A.Resize(image_size, image_size),
            
            # Augmentations → le modèle voit des variations
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            
            # Normalisation standard ImageNet
            # (obligatoire avec modèle pré-entraîné)
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            
            # Convertir numpy array → tensor PyTorch
            ToTensorV2()
        ])
    
    else:  # val ou test
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

# ============================================
# CLASSE DATASET
# ============================================
class AerospaceDataset(Dataset):
    """
    Lit les images depuis :
    data/
      train/
        classe_A/  image1.jpg  image2.jpg ...
        classe_B/  image1.jpg  ...
      val/
        ...
      test/
        ...
    """
    
    def __init__(self, root_dir, mode='train', image_size=224):
        self.root_dir  = Path(root_dir)
        self.mode      = mode
        self.transform = get_transforms(mode, image_size)
        
        # Trouver toutes les classes (= noms des sous-dossiers)
        split_dir   = self.root_dir / mode
        self.classes = sorted([
            d.name for d in split_dir.iterdir() 
            if d.is_dir()
        ])
        
        # Dictionnaire : nom de classe → numéro
        # Ex : {'airplane': 0, 'car': 1, ...}
        self.class_to_idx = {
            cls: i for i, cls in enumerate(self.classes)
        }
        
        # Liste de toutes les images avec leur label
        # Ex : [('data/train/airplane/img1.jpg', 0), ...]
        self.samples = []
        for cls in self.classes:
            cls_dir = split_dir / cls
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                for img_path in cls_dir.glob(ext):
                    self.samples.append(
                        (img_path, self.class_to_idx[cls])
                    )
        
        print(f"✅ {mode:5s}: {len(self.samples):5d} images "
              f"| {len(self.classes)} classes")
    
    def __len__(self):
        # PyTorch a besoin de connaître la taille
        return len(self.samples)
    
    def __getitem__(self, idx):
        # PyTorch appelle cette fonction pour chaque image
        img_path, label = self.samples[idx]
        
        # Ouvrir l'image
        image = np.array(
            Image.open(img_path).convert('RGB')
        )
        
        # Appliquer les transformations
        if self.transform:
            augmented = self.transform(image=image)
            image     = augmented['image']
        
        return image, label  # tensor, int


# ============================================
# CRÉER LES DATALOADERS
# ============================================
def get_dataloaders(config_path='configs/config.yaml'):
    """
    Retourne les 3 dataloaders : train, val, test
    Le DataLoader charge les images en mini-batches
    """
    cfg        = load_config(config_path)
    image_size = cfg['model']['image_size']
    batch_size = cfg['training']['batch_size']
    num_workers= cfg['data']['num_workers']
    
    # Créer les 3 datasets
    train_ds = AerospaceDataset('data', 'train', image_size)
    val_ds   = AerospaceDataset('data', 'val',   image_size)
    test_ds  = AerospaceDataset('data', 'test',  image_size)
    
    # Créer les DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,          # Mélanger à chaque epoch
        num_workers=num_workers,
        pin_memory=True        # Plus rapide sur GPU
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,         # Pas besoin de mélanger
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader, train_ds.classes


# ============================================
# TEST RAPIDE
# ============================================
if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders()
    
    print(f"\nClasses : {classes}")
    
    # Vérifier une batch
    images, labels = next(iter(train_loader))
    print(f"Shape batch images : {images.shape}")
    print(f"Shape batch labels : {labels.shape}")
    # Attendu : torch.Size([32, 3, 224, 224])
    #           torch.Size([32])
