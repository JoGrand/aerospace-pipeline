# api/main.py
# API de production avec FastAPI

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image
import io
import time
import yaml

from src.model import AerospaceClassifier
from src.dataset import get_transforms

# ============================================
# VARIABLES GLOBALES
# ============================================
model     = None
classes   = None
transform = None
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================
# CHARGEMENT DU MODÈLE
# ============================================
def load_model():
    global model, classes, transform

    print("📂 Chargement du modèle...")

    try:
        checkpoint = torch.load(
            'models/best_model.pth',
            map_location=device
        )

        classes    = checkpoint['classes']
        cfg        = checkpoint['config']
        image_size = cfg['model']['image_size']

        model = AerospaceClassifier(
            num_classes=len(classes),
            pretrained=False
        )
        model.load_state_dict(checkpoint['model_state'])
        model.to(device)
        model.eval()

        transform = get_transforms('val', image_size)

        print(f"✅ Modèle chargé avec succès sur {device}")
        print(f"📋 Classes : {classes}")

    except FileNotFoundError:
        print("⚠️  Aucun modèle trouvé — lancer src/train.py d'abord")

# ============================================
# LIFESPAN (démarre/arrête avec l'app)
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Démarrage
    load_model()
    yield
    # Arrêt
    print("🛑 API arrêtée")

# ============================================
# CRÉER L'APPLICATION
# ============================================
app = FastAPI(
    title="🛰️ Aerospace CV API",
    description="Classification d'images par Deep Learning",
    version="1.0.0",
    lifespan=lifespan
)

# Autoriser les requêtes depuis n'importe où
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# ROUTES
# ============================================
@app.get("/")
async def root():
    """Page d'accueil"""
    return {
        "message" : "🛰️ Aerospace CV API",
        "status"  : "running",
        "docs"    : "/docs"
    }

@app.get("/health")
async def health():
    """Vérifier que l'API et le modèle sont prêts"""
    return {
        "status"       : "healthy" if model is not None else "no model loaded",
        "model_loaded" : model is not None,
        "device"       : str(device),
        "classes"      : classes
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Classifier une image
    - **file** : image JPG/PNG
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non chargé — lancer src/train.py d'abord"
        )

    # Vérifier le type de fichier
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté : {file.content_type}. Utiliser JPG ou PNG"
        )

    try:
        # Lire et préparer l'image
        contents = await file.read()
        image    = Image.open(io.BytesIO(contents)).convert('RGB')
        tensor   = transform(image).unsqueeze(0).to(device)

        # Inférence
        start_time = time.time()

        with torch.no_grad():
            outputs     = model(tensor)
            probs       = torch.softmax(outputs, dim=1)[0]
            pred_idx    = torch.argmax(probs).item()
            confidence  = probs[pred_idx].item()

        inference_time = (time.time() - start_time) * 1000  # ms

        # Toutes les probabilités
        all_probs = {
            cls: round(probs[i].item(), 4)
            for i, cls in enumerate(classes)
        }

        return {
            "prediction"     : classes[pred_idx],
            "confidence"     : round(confidence, 4),
            "all_classes"    : all_probs,
            "inference_ms"   : round(inference_time, 2),
            "filename"       : file.filename
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction : {str(e)}"
        )

@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Classifier plusieurs images en une seule requête"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images par batch")

    results = []

    for file in files:
        try:
            contents = await file.read()
            image    = Image.open(io.BytesIO(contents)).convert('RGB')
            tensor   = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs    = model(tensor)
                probs      = torch.softmax(outputs, dim=1)[0]
                pred_idx   = torch.argmax(probs).item()
                confidence = probs[pred_idx].item()

            results.append({
                "filename"   : file.filename,
                "prediction" : classes[pred_idx],
                "confidence" : round(confidence, 4),
                "status"     : "success"
            })

        except Exception as e:
            results.append({
                "filename" : file.filename,
                "status"   : "error",
                "detail"   : str(e)
            })

    return {
        "total"   : len(files),
        "results" : results
    }
