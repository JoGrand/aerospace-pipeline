# src/evaluate.py
# Tester le modèle sur des données jamais vues

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import os
import sys

from dataset import get_dataloaders
from model import AerospaceClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model(model_path='models/best_model.pth'):
    
    print("\n📂 Chargement du modèle...")
    
    # Recharger le checkpoint sauvegardé
    checkpoint  = torch.load(model_path, map_location=device)
    classes     = checkpoint['classes']
    cfg         = checkpoint['config']
    num_classes = len(classes)
    
    # Recréer le modèle avec la même architecture
    model = AerospaceClassifier(
        num_classes=num_classes,
        pretrained=False    # Pas besoin de retélécharger
    )
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Modèle chargé !")
    print(f"   Val acc entraînement : {checkpoint['val_acc']:.2f}%")
    
    # Charger les données de test
    _, _, test_loader, _ = get_dataloaders()
    
    # ---- FAIRE LES PRÉDICTIONS ----
    print("\n🔍 Évaluation sur le test set...")
    
    all_preds  = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images  = images.to(device)
            outputs = model(images)
            
            # Prendre la classe avec le score le plus haut
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # ---- MÉTRIQUES ----
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n🎯 Test Accuracy : {accuracy*100:.2f}%")
    
    # Rapport détaillé par classe
    print("\n📊 Rapport par classe :")
    report = classification_report(
        all_labels, all_preds,
        target_names=classes
    )
    print(report)
    
    # Sauvegarder le rapport
    os.makedirs('results', exist_ok=True)
    with open('results/classification_report.txt', 'w') as f:
        f.write(f"Test Accuracy : {accuracy*100:.2f}%\n\n")
        f.write(report)
    
    # ---- MATRICE DE CONFUSION ----
    plot_confusion_matrix(all_labels, all_preds, classes)
    
    # ---- COURBES D'ENTRAÎNEMENT ----
    plot_training_curves()
    
    return accuracy


def plot_confusion_matrix(labels, preds, classes):
    """Visualise quelles classes sont confondues"""
    
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.0%',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        ax=ax
    )
    
    ax.set_xlabel('Prédit',  fontsize=12)
    ax.set_ylabel('Réel',    fontsize=12)
    ax.set_title('Matrice de Confusion', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=150)
    plt.show()
    
    print("✅ Matrice sauvegardée → results/confusion_matrix.png")


def plot_training_curves():
    """Affiche les courbes loss et accuracy"""
    
    import json
    
    if not os.path.exists('results/history.json'):
        print("⚠️  Pas d'historique trouvé")
        return
    
    with open('results/history.json') as f:
        history = json.load(f)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Courbe Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax1.plot(epochs, history['val_loss'],   'r-', label='Val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Courbe Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train')
    ax2.plot(epochs, history['val_acc'],   'r-', label='Val')
    ax2.set_title('Accuracy (%)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Courbes d\'entraînement', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=150)
    plt.show()
    
    print("✅ Courbes sauvegardées → results/training_curves.png")


if __name__ == "__main__":
    evaluate_model()
