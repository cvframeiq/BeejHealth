"""
╔══════════════════════════════════════════════════════════════╗
║   BeejHealth — Coconut Disease AI Server                    ║
║   Model: EfficientNetV2-S (Transfer Learning)               ║
║   8 Classes | Port: 8000                                    ║
║   Run: python ai_server.py                                  ║
╚══════════════════════════════════════════════════════════════╝
"""

import os, io, base64, time, logging, traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

# ── Logging ──────────────────────────────────────────────────────────
import sys
os.makedirs("logs", exist_ok=True)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_stream_handler = logging.StreamHandler(stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1))
_stream_handler.setFormatter(_fmt)
_file_handler = logging.FileHandler("logs/ai_server.log", encoding="utf-8")
_file_handler.setFormatter(_fmt)
logging.basicConfig(level=logging.INFO, handlers=[_stream_handler, _file_handler])
log = logging.getLogger("coconut-ai")

# ── Config ────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/Best_coconut_model.pth")
AI_PORT    = int(os.getenv("AI_PORT", 8000))
MIN_CONF   = float(os.getenv("MIN_CONFIDENCE", 0.30))
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 8 Classes — EXACT sorted order as CoconutDiseaseDataset ──────────
CLASS_NAMES = [
    "Gray Leaf Spot",
    "Gray Leaf Spot_multiple",
    "Healthy",
    "Leaf rot",
    "Leaf rot_multiple",
    "bud rot",
    "stem bleeding",
    "stem bleeding_multiple",
]

# ── Disease metadata ──────────────────────────────────────────────────
DISEASE_INFO = {
    "Gray Leaf Spot": {
        "hindi": "Dhusrit Patti Dhabb", "severity": 2, "is_healthy": False,
        "description": "Patti pe gray/brown chote daag. Pestalotiopsis fungus se hota hai.",
        "urgency": "medium",
        "treatments": [
            "Copper Oxychloride 3g/L spray karein",
            "Affected patte turant hatao aur jalao",
            "Mancozeb 75% WP (2.5g/L) 15 din mein 2 baar",
            "Overwatering avoid karein",
        ],
        "spread_risk": "Baarish mein 40% tak badh sakta hai",
        "recovery": "7-14 din mein sudhaar sahi treatment se",
    },
    "Gray Leaf Spot_multiple": {
        "hindi": "Bahut Dhusrit Patti Dhabb", "severity": 3, "is_healthy": False,
        "description": "Kai pattiyon pe gray spots — infection failne lagi hai.",
        "urgency": "high",
        "treatments": [
            "Turant Copper-based fungicide spray karein",
            "Saari affected leaves kaat ke jalao",
            "Propiconazole 1ml/L spray 10 din interval pe",
            "Tree ke around drainage improve karein",
        ],
        "spread_risk": "60% aas-paas ke plants mein ja sakta hai",
        "recovery": "14-21 din mein sudhaar treatment se",
    },
    "Healthy": {
        "hindi": "Swasth Naariyal Ped", "severity": 0, "is_healthy": True,
        "description": "Aapka coconut tree bilkul healthy hai! Koi bimari nahi mili.",
        "urgency": "none",
        "treatments": [
            "Regular monthly monitoring jaari rakhein",
            "NPK fertilizer schedule follow karein",
            "Proper irrigation maintain karein",
        ],
        "spread_risk": "Koi risk nahi",
        "recovery": "Koi ilaaj zaroorat nahi",
    },
    "Leaf rot": {
        "hindi": "Patti Galaav", "severity": 3, "is_healthy": False,
        "description": "Patte galane lag rahe hain. Colletotrichum fungus, nami se hota hai.",
        "urgency": "high",
        "treatments": [
            "Bordeaux mixture 1% spray karein",
            "Rogi pattiyan immediately remove karein",
            "Overwatering bilkul band karein",
            "Carbendazim 1g/L fortnightly spray",
        ],
        "spread_risk": "Nami mein 70% tak fail sakta hai",
        "recovery": "10-20 din mein theek ho sakta hai",
    },
    "Leaf rot_multiple": {
        "hindi": "Bahut Patti Galaav", "severity": 4, "is_healthy": False,
        "description": "Kai pattiyon mein rot — serious infection, turant action zaroor.",
        "urgency": "critical",
        "treatments": [
            "EMERGENCY: Turant expert se milein",
            "Saari rogi pattiyan kaat ke jalao",
            "Hexaconazole 2ml/L spray immediately",
            "Tree ke around proper drainage banao",
        ],
        "spread_risk": "Pure tree ko 2 hafte mein affect kar sakta hai",
        "recovery": "3-4 hafte expert treatment se",
    },
    "bud rot": {
        "hindi": "Kali Galaav", "severity": 5, "is_healthy": False,
        "description": "Coconut ki nai kali gal rahi hai. Phytophthora palmivora — bahut khatarnak!",
        "urgency": "critical",
        "treatments": [
            "CRITICAL: Expert consultation turant zaroor",
            "Metalaxyl + Mancozeb 3g/L drench apply karein",
            "Affected bud tissue carefully remove karein",
            "Tree ko isolated karein infection rokne ke liye",
            "Potassium fertilizer dein immunity ke liye",
        ],
        "spread_risk": "Ek ped se doosre mein turant ja sakta hai",
        "recovery": "Early stage mein 60% recovery possible",
    },
    "stem bleeding": {
        "hindi": "Tane se Khoon", "severity": 4, "is_healthy": False,
        "description": "Trunk se brown/red liquid nikalna. Thielaviopsis paradoxa fungus se.",
        "urgency": "high",
        "treatments": [
            "Bleeding area ko chisel se saaf karein",
            "Bordeaux paste wound pe lagao",
            "Carbendazim 2g/L trunk injection lagao",
            "Affected area pe tar coal apply karein",
        ],
        "spread_risk": "Soil mein spread ho sakta hai",
        "recovery": "2-3 mahine mein sudhaar aata hai",
    },
    "stem bleeding_multiple": {
        "hindi": "Bahut Tane se Khoon", "severity": 5, "is_healthy": False,
        "description": "Multiple jagah bleeding — advanced infection hai.",
        "urgency": "critical",
        "treatments": [
            "EMERGENCY: Agricultural officer ko turant bulao",
            "Saari bleeding sites simultaneously treat karein",
            "Trichoderma soil application karein",
            "Systemic fungicide trunk injection",
        ],
        "spread_risk": "Pure farm mein fail sakta hai",
        "recovery": "Uncertain — expert guidance zaroor",
    },
}

# ── Model Architecture — EXACT from notebook Cell 8 ──────────────────
class CoconutDiseaseClassifier(nn.Module):
    def __init__(self, num_classes=8, model_name="tf_efficientnetv2_s"):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.model(x)
        return self.classifier(features)


# ── Preprocessing — EXACT val_transform from notebook Cell 6 ─────────
val_transform = A.Compose([
    A.Resize(384, 384),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ── Global model ──────────────────────────────────────────────────────
model = None
class_names_loaded = CLASS_NAMES

def load_model():
    global model, class_names_loaded
    log.info(f"Loading model: {MODEL_PATH} | Device: {DEVICE}")

    if not Path(MODEL_PATH).exists():
        log.error(f"❌ Model NOT FOUND: {MODEL_PATH}")
        log.error("   ai_service/models/ folder mein Best_Cocunut_model.pth rakhein")
        return False
    try:
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(ckpt, dict) and "class_names" in ckpt:
            class_names_loaded = ckpt["class_names"]
        num_classes = len(class_names_loaded)
        model = CoconutDiseaseClassifier(num_classes=num_classes).to(DEVICE)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()
        # Warmup
        with torch.no_grad():
            model(torch.randn(1, 3, 384, 384).to(DEVICE))
        log.info(f"✅ Model ready! {num_classes} classes | {DEVICE}")
        return True
    except Exception as e:
        log.error(f"Model load failed: {e}\n{traceback.format_exc()}")
        return False


# ── FastAPI ───────────────────────────────────────────────────────────
app = FastAPI(
    title="BeejHealth Coconut AI",
    description="EfficientNetV2-S — Coconut Disease Detection",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_methods=["*"], allow_headers=["*"],
)


class PredictRequest(BaseModel):
    image_base64: str
    consultation_id: Optional[str] = None
    question_answers: Optional[dict] = None


def run_inference(img_np: np.ndarray) -> dict:
    t0 = time.time()
    transformed = val_transform(image=img_np)
    tensor = transformed["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)
    top3_conf, top3_idx = torch.topk(probs, min(3, len(class_names_loaded)))
    top3 = []
    for i in range(top3_conf.shape[1]):
        idx  = top3_idx[0][i].item()
        conf = top3_conf[0][i].item() * 100
        nm   = class_names_loaded[idx] if idx < len(class_names_loaded) else f"class_{idx}"
        info = DISEASE_INFO.get(nm, {})
        top3.append({"rank": i+1, "disease": nm, "confidence": round(conf, 2), "severity": info.get("severity", 0)})
    best = top3[0]
    info = DISEASE_INFO.get(best["disease"], DISEASE_INFO["Healthy"])
    elapsed = (time.time() - t0) * 1000
    log.info(f"Prediction: {best['disease']} ({best['confidence']:.1f}%) in {elapsed:.0f}ms")
    return {
        "success": True,
        "disease": best["disease"],
        "disease_hindi": info.get("hindi", best["disease"]),
        "confidence": best["confidence"],
        "severity": info.get("severity", 0),
        "is_healthy": info.get("is_healthy", False),
        "urgency": info.get("urgency", "medium"),
        "description": info.get("description", ""),
        "treatments": info.get("treatments", []),
        "spread_risk": info.get("spread_risk", ""),
        "recovery": info.get("recovery", ""),
        "top3": top3,
        "processing_time_ms": round(elapsed, 1),
        "model_version": "EfficientNetV2-S-v1.0",
        "coconut_only": True,
    }


@app.get("/")
def root():
    return {"app": "BeejHealth Coconut AI", "status": "ready" if model else "model_not_loaded", "classes": CLASS_NAMES}

@app.get("/health")
def health():
    return {"status": "ok" if model else "model_not_loaded", "device": str(DEVICE), "model_loaded": model is not None, "num_classes": len(class_names_loaded)}

@app.post("/predict")
async def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(503, "Model load nahi hua. models/Best_Cocunut_model.pth rakhein.")
    try:
        b64 = req.image_base64
        if "," in b64:
            b64 = b64.split(",")[1]
        img_bytes = base64.b64decode(b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(pil_img)
        result = run_inference(img_np)
        if req.question_answers:
            result["question_answers"] = req.question_answers
        return result
    except Exception as e:
        log.error(f"Predict error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))

@app.post("/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Model load nahi hua.")
    content = await file.read()
    img_np = np.array(Image.open(io.BytesIO(content)).convert("RGB"))
    return run_inference(img_np)

@app.get("/classes")
def get_classes():
    return {"classes": class_names_loaded, "num_classes": len(class_names_loaded),
            "disease_info": {n: {"hindi": DISEASE_INFO[n]["hindi"], "severity": DISEASE_INFO[n]["severity"], "urgency": DISEASE_INFO[n]["urgency"]} for n in CLASS_NAMES if n in DISEASE_INFO}}

@app.on_event("startup")
def startup():
    log.info("=" * 55)
    log.info("  BeejHealth Coconut AI Server Starting...")
    log.info("=" * 55)
    ok = load_model()
    if not ok:
        log.warning("⚠️  Model load nahi hua! models/ folder check karein.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai_server:app", host="0.0.0.0", port=AI_PORT, reload=False)
