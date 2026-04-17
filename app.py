from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2

app = FastAPI()

@app.get("/")
def home():
    return {"message": "ML Service Running"}

@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    contents = await image.read()
    
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image"}

    # Dummy logic (replace later with real model)
    return {
        "face_detected": True,
        "liveness": True,
        "head_movement": "center"
    }