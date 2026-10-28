from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import numpy as np
from tensorflow.keras.preprocessing import image

from backend.model import predict_image, model
from backend.gradcam import generate_gradcam
from backend.database import init_db, save_prediction, get_history, search_history_by_name

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, '..', 'static', 'uploads')
GRADCAM_DIR = os.path.join(BASE_DIR, '..', 'static', 'gradcam')
STATIC_DIR = os.path.join(BASE_DIR, '..', 'static')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize SQLite DB on startup/import
init_db()


@app.get("/")
def home():
    return {"message": "Pneumonia Detection API is running"}


@app.post("/predict")
async def predict(
    patient_name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    notes: str = Form(""),
    file: UploadFile = File(...)
):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        label, confidence = predict_image(file_path)

        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        gradcam_filename = "grad_" + file.filename
        gradcam_path = os.path.join(GRADCAM_DIR, gradcam_filename)

        generate_gradcam(model, img_array, file_path, gradcam_path)

        save_prediction(
            patient_name=patient_name,
            age=age,
            gender=gender,
            notes=notes,
            filename=file.filename,
            prediction=label,
            confidence=float(confidence),
            gradcam_image=gradcam_filename
        )

        return {
            "patient_name": patient_name,
            "age": age,
            "gender": gender,
            "notes": notes,
            "prediction": label,
            "confidence": round(float(confidence), 4),
            "gradcam_image": gradcam_filename
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/history")
def history():
    rows = get_history()
    results = []

    for row in rows:
        results.append({
            "id": row[0],
            "patient_name": row[1],
            "age": row[2],
            "gender": row[3],
            "notes": row[4],
            "filename": row[5],
            "prediction": row[6],
            "confidence": row[7],
            "gradcam_image": row[8],
            "created_at": row[9]
        })

    return results


@app.get("/history/search")
def search_history(name: str):
    rows = search_history_by_name(name)
    results = []

    for row in rows:
        results.append({
            "id": row[0],
            "patient_name": row[1],
            "age": row[2],
            "gender": row[3],
            "notes": row[4],
            "filename": row[5],
            "prediction": row[6],
            "confidence": row[7],
            "gradcam_image": row[8],
            "created_at": row[9]
        })

    return results