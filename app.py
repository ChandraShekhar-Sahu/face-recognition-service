from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
import traceback

app = FastAPI()

# Load Haar Cascade once
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
# Add this near the top of your file, outside of the endpoints so they only load once
detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (300, 300))
recognizer = cv2.FaceRecognizerSF.create("face_recognition_sface_2021dec.onnx", "")


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

        # -------- FACE DETECTION --------
        gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        gray_cur = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)

        ref_faces = face_cascade.detectMultiScale(gray_ref, 1.1, 3)
        cur_faces = face_cascade.detectMultiScale(gray_cur, 1.1, 3)

        if len(ref_faces) == 0 or len(cur_faces) == 0:
            return {
                "match": False,
                "face_detected": False,
                "message": "No face detected"
            }

        # -------- SIMPLE MATCH (TEMP) --------
        ref_img = cv2.resize(ref_img, (300, 300))
        cur_img = cv2.resize(cur_img, (300, 300))

        diff = np.mean(cv2.absdiff(ref_img, cur_img))
        match = diff < 50

        # -------- HEAD MOVEMENT (SIMPLIFIED) --------
        head_movement = "normal"

        # -------- LIVENESS (SIMPLIFIED) --------
        liveness = True
        liveness_text = "Basic detection - assumed live"

        return {
            "match": bool(match),
            "face_detected": True,
            "message": "Face Matched!" if match else "Face Not Matched!",
            "liveness": liveness,
            "liveness_text": liveness_text,
            "head_movement": head_movement
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
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