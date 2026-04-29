# src/model.py
# Architecture du réseau de neurones

import torch
import torch.nn as nn
import timm    # librairie avec des centaines de modèles SOTA
import yaml

def load_config(path='configs/config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

# ============================================
# ARCHITECTURE
# ============================================
class AerospaceClassifier(nn.Module):
    """
    EfficientNetV2 = réseau pré-entraîné sur ImageNet
    On remplace juste la dernière couche
    pour nos propres classes
    
    Analogie :
    - Backbone = cerveau pré-formé (reconnaît formes, textures)
    - Head     = spécialisation pour notre tâche
    """
    
    def __init__(self, num_classes=10, 
                 pretrained=True, dropout=0.3):
        super().__init__()
        
        # ---- BACKBONE ----
        # efficientnetv2_rw_s = version "small" (plus rapide)
        # Pour débutant : commencer avec la version small
        self.backbone = timm.create_model(
            'efficientnetv2_rw_s',
            pretrained=pretrained,
            num_classes=0,       # 0 = enlève la dernière couche
            global_pool='avg'    # Moyenne spatiale des features
        )
        
        # Taille de sortie du backbone
        n_features = self.backbone.num_features  # 1792
        
        # ---- HEAD (tête de classification) ----
        self.head = nn.Sequential(
            nn.LayerNorm(n_features),    # Normalisation
            nn.Dropout(dropout),          # Régularisation
            nn.Linear(n_features, 512),  # Couche dense
            nn.GELU(),                    # Activation
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)  # Sortie finale
        )
        
        # Afficher infos
        total_params = sum(
            p.numel() for p in self.parameters()
        ) / 1e6
        print(f"🧠 Modèle créé")
        print(f"   Features backbone : {n_features}")
        print(f"   Nombre de classes : {num_classes}")
        print(f"   Paramètres totaux : {total_params:.1f}M")
    
    def forward(self, x):
        """
        x → backbone → features → head → logits
        """
        features = self.backbone(x)   # Extraire features
        logits   = self.head(features) # Classer
        return logits
    
    def freeze_backbone(self):
        """
        Geler le backbone = ne pas modifier ses poids
        Phase 1 : entraîner seulement la tête
        → Plus rapide, moins de risque d'overfitting
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        trainable = sum(
            p.numel() for p in self.parameters() 
            if p.requires_grad
        ) / 1e6
        print(f"🔒 Backbone gelé | Paramètres entraînables : {trainable:.1f}M")
    
    def unfreeze_backbone(self):
        """
        Dégeler = fine-tuner tout le réseau
        Phase 2 : affiner avec un très petit learning rate
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        trainable = sum(
            p.numel() for p in self.parameters() 
            if p.requires_grad
        ) / 1e6
        print(f"🔓 Backbone dégelé | Paramètres entraînables : {trainable:.1f}M")


def create_model(config_path='configs/config.yaml'):
    cfg = load_config(config_path)
    return AerospaceClassifier(
        num_classes=cfg['model']['num_classes'],
        pretrained=cfg['model']['pretrained']
    )


# Test rapide
if __name__ == "__main__":
    model = create_model()
    
    # Simuler une batch de 2 images
    dummy_input = torch.randn(2, 3, 224, 224)
    output      = model(dummy_input)
    
    print(f"\nTest forward pass :")
    print(f"   Input  : {dummy_input.shape}")
    print(f"   Output : {output.shape}")
    # Attendu : torch.Size([2, 10])
