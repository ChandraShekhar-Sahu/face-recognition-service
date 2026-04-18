from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import traceback

app = FastAPI()

# Load Haar Cascade once
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@app.post("/verify-face")
async def verify_face(
    reference: UploadFile = File(...),
    current: UploadFile = File(...)
):
    try:
        ref_bytes = await reference.read()
        cur_bytes = await current.read()

        ref_arr = np.frombuffer(ref_bytes, np.uint8)
        cur_arr = np.frombuffer(cur_bytes, np.uint8)

        ref_img = cv2.imdecode(ref_arr, cv2.IMREAD_COLOR)
        cur_img = cv2.imdecode(cur_arr, cv2.IMREAD_COLOR)

        if ref_img is None or cur_img is None:
            return {"error": "Image decode failed"}

        ref_img = cv2.resize(ref_img, (300, 300))
        cur_img = cv2.resize(cur_img, (300, 300))

        diff = np.mean(cv2.absdiff(ref_img, cur_img))

        return {
            # FIX: Convert numpy bool to native Python bool
            "match": bool(diff < 50),
            "difference": float(diff)
        }

    except Exception as e:
        print("ERROR OCCURRED:")
        traceback.print_exc()

        return {
            "error": str(e)
        }

def get_face_embedding(face):
    face = cv2.resize(face, (100, 100))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    return face.flatten() / 255.0


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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 🔥 enhance image
    gray = cv2.equalizeHist(gray)

    # 🔥 primary detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30)
    )

    # 🔥 fallback detection
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(20, 20)
        )

    face_detected = len(faces) > 0

    # FIX: Removed the duplicate image decoding block that was pasted down here

    return {
        "face_detected": face_detected,
        "liveness": face_detected,
        "head_movement": "center" if face_detected else "unknown"
    }