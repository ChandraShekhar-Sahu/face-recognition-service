from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2

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
    ref_bytes = await reference.read()
    cur_bytes = await current.read()

    ref_np = np.frombuffer(ref_bytes, np.uint8)
    cur_np = np.frombuffer(cur_bytes, np.uint8)

    ref_img = cv2.imdecode(ref_np, cv2.IMREAD_COLOR)
    cur_img = cv2.imdecode(cur_np, cv2.IMREAD_COLOR)

    if ref_img is None or cur_img is None:
        return {"error": "Invalid image"}

    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    gray_cur = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)

    faces_ref = face_cascade.detectMultiScale(gray_ref, 1.1, 3)
    faces_cur = face_cascade.detectMultiScale(gray_cur, 1.1, 3)

    if len(faces_ref) == 0 or len(faces_cur) == 0:
        return {"match": False, "message": "Face not detected"}

    x,y,w,h = faces_ref[0]
    ref_face = ref_img[y:y+h, x:x+w]

    x,y,w,h = faces_cur[0]
    cur_face = cur_img[y:y+h, x:x+w]

    emb1 = get_face_embedding(ref_face)
    emb2 = get_face_embedding(cur_face)

    distance = np.linalg.norm(emb1 - emb2)

    return {
        "match": distance < 0.6,
        "distance": float(distance)
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

    return {
        "face_detected": face_detected,
        "liveness": face_detected,
        "head_movement": "center" if face_detected else "unknown"
    }
    contents = await image.read()
    
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image"}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30)
    )

    # fallback if no face found
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=2,
            minSize=(20, 20)
        )

    face_detected = len(faces) > 0

    return {
        "face_detected": face_detected,
        "liveness": face_detected,  # temporary
        "head_movement": "center" if face_detected else "unknown"
    }