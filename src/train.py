# src/train.py
# Boucle d'entraînement principale

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import mlflow
import yaml
import time
import json
import os
from tqdm import tqdm

from dataset import get_dataloaders
from model import create_model

def load_config(path='configs/config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)

# ============================================
# UNE EPOCH D'ENTRAÎNEMENT
# ============================================
def train_one_epoch(model, loader, criterion,
                    optimizer, scaler, device):
    """
    Une epoch = le modèle voit TOUTES les images une fois
    """
    model.train()  # Mode entraînement (active dropout etc.)
    
    total_loss = 0.0
    correct    = 0
    total      = 0
    
    # tqdm = barre de progression
    pbar = tqdm(loader, desc='  Train', leave=False)
    
    for images, labels in pbar:
        # Envoyer sur GPU si disponible
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Remettre les gradients à zéro
        optimizer.zero_grad()
        
        # ---- FORWARD PASS (mixe précision = plus rapide) ----
        with autocast():
            outputs = model(images)        # Prédictions
            loss    = criterion(outputs, labels)  # Erreur
        
        # ---- BACKWARD PASS (calculer gradients) ----
        scaler.scale(loss).backward()
        
        # Éviter les gradients explosifs
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )
        
        # ---- MISE À JOUR DES POIDS ----
        scaler.step(optimizer)
        scaler.update()
        
        # Statistiques
        total_loss  += loss.item()
        _, predicted = outputs.max(1)
        total       += labels.size(0)
        correct     += predicted.eq(labels).sum().item()
        
        # Mettre à jour la barre
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc' : f'{100.*correct/total:.1f}%'
        })
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


# ============================================
# VALIDATION
# ============================================
@torch.no_grad()  # Pas de gradients = plus rapide
def validate(model, loader, criterion, device):
    model.eval()  # Mode évaluation (désactive dropout)
    
    total_loss = 0.0
    correct    = 0
    total      = 0
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast():
            outputs = model(images)
            loss    = criterion(outputs, labels)
        
        total_loss  += loss.item()
        _, predicted = outputs.max(1)
        total       += labels.size(0)
        correct     += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


# ============================================
# BOUCLE PRINCIPALE
# ============================================
def train():
    cfg = load_config()

    # Pour mixed precision
    scaler = torch.amp.GradScaler('cuda')
    
    # GPU ou CPU ?
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"\n💻 Device : {device}")
    if device.type == 'cpu':
        print("   ⚠️  Pas de GPU détecté → entraînement lent")
        print("   💡 Réduire epochs à 3 pour tester\n")
        scaler = torch.GradScaler()
    
    # ---- DONNÉES ----
    train_loader, val_loader, _, classes = get_dataloaders()
    
    # ---- MODÈLE ----
    model = create_model().to(device)
    
    # Phase 1 : backbone gelé
    model.freeze_backbone()
    
    # ---- LOSS ----
    # label_smoothing=0.1 → évite la sur-confiance
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # ---- OPTIMIZER ----
    # AdamW = Adam amélioré, très standard
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay']
    )
    
    # ---- SCHEDULER ----
    # Réduit progressivement le learning rate
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['training']['epochs']
    )
    
    # ---- VARIABLES DE SUIVI ----
    best_val_acc   = 0.0
    patience_count = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc' : [], 'val_acc' : []
    }
    
    os.makedirs('models',  exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # ==========================================
    # MLFLOW : tracker l'expérience
    # ==========================================
    mlflow.set_tracking_uri(cfg['mlops']['tracking_uri'])
    mlflow.set_experiment(cfg['mlops']['experiment_name'])
    
    print("\n" + "="*60)
    print("🚀 DÉBUT DE L'ENTRAÎNEMENT")
    print("="*60 + "\n")
    
    with mlflow.start_run():
        
        # Sauvegarder les hyperparamètres dans MLflow
        mlflow.log_params({
            'model'        : cfg['model']['name'],
            'epochs'       : cfg['training']['epochs'],
            'lr'           : cfg['training']['learning_rate'],
            'batch_size'   : cfg['training']['batch_size'],
            'num_classes'  : cfg['model']['num_classes'],
        })
        
        # ======================================
        # BOUCLE SUR LES EPOCHS
        # ======================================
        for epoch in range(cfg['training']['epochs']):
            
            # À mi-chemin : dégeler le backbone (fine-tuning)
            halfway = cfg['training']['epochs'] // 2
            if epoch == halfway:
                print(f"\n{'='*40}")
                print(f"🔓 Phase 2 : Fine-tuning complet")
                print(f"{'='*40}\n")
                model.unfreeze_backbone()
                # Baisser beaucoup le LR pour ne pas "oublier"
                for g in optimizer.param_groups:
                    g['lr'] = cfg['training']['learning_rate'] * 0.1
            
            t_start = time.time()
            
            # Entraîner
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion,
                optimizer, scaler, device
            )
            # Valider
            val_loss, val_acc = validate(
                model, val_loader, criterion, device
            )
            
            scheduler.step()
            elapsed = time.time() - t_start
            
            # Afficher résultats
            print(
                f"Epoch {epoch+1:3d}/{cfg['training']['epochs']} "
                f"│ Train {train_acc:5.1f}% (loss {train_loss:.3f}) "
                f"│ Val {val_acc:5.1f}% (loss {val_loss:.3f}) "
                f"│ {elapsed:.0f}s"
            )
            
            # Log dans MLflow
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_acc' : train_acc,
                'val_loss'  : val_loss,
                'val_acc'   : val_acc,
            }, step=epoch)
            
            # Sauvegarder dans l'historique
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # ---- SAUVEGARDER LE MEILLEUR MODÈLE ----
            if val_acc > best_val_acc:
                best_val_acc   = val_acc
                patience_count = 0
                
                torch.save({
                    'epoch'      : epoch,
                    'model_state': model.state_dict(),
                    'val_acc'    : val_acc,
                    'classes'    : classes,
                    'config'     : cfg,
                }, 'models/best_model.pth')
                
                print(f"   💾 Nouveau meilleur modèle : {val_acc:.2f}%")
            
            else:
                patience_count += 1
                print(f"   ⏳ Patience : {patience_count}/{cfg['training']['patience']}")
                
                # Arrêter si pas d'amélioration
                if patience_count >= cfg['training']['patience']:
                    print(f"\n⏹️  Early stopping !")
                    break
        
        # Log meilleur résultat
        mlflow.log_metric('best_val_acc', best_val_acc)
    
    # Sauvegarder l'historique
    with open('results/history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Entraînement terminé !")
    print(f"🏆 Meilleure Val Accuracy : {best_val_acc:.2f}%")
    print(f"💾 Modèle sauvegardé dans models/best_model.pth")
    print(f"{'='*60}")
    
    return history


if __name__ == "__main__":
    train()
