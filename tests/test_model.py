# tests/test_model.py

import pytest
import torch
from src.model import AerospaceClassifier

class TestModel:
    
    def test_creation(self):
        """Le modèle se crée sans erreur"""
        model = AerospaceClassifier(num_classes=5, pretrained=False)
        assert model is not None
    
    def test_output_shape(self):
        """La sortie a la bonne forme"""
        model = AerospaceClassifier(num_classes=5, pretrained=False)
        model.eval()
        
        dummy  = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy)
        
        assert output.shape == (2, 5), \
            f"Attendu (2,5), obtenu {output.shape}"
    
    def test_freeze(self):
        """Le gel du backbone fonctionne"""
        model = AerospaceClassifier(num_classes=5, pretrained=False)
        model.freeze_backbone()
        
        for param in model.backbone.parameters():
            assert not param.requires_grad
    
    def test_unfreeze(self):
        """Le dégel du backbone fonctionne"""
        model = AerospaceClassifier(num_classes=5, pretrained=False)
        model.freeze_backbone()
        model.unfreeze_backbone()
        
        for param in model.backbone.parameters():
            assert param.requires_grad
