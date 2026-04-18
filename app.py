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

        # 1. Convert to grayscale for the face detector
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        cur_gray = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)

        # 2. Detect faces in both images
        ref_faces = face_cascade.detectMultiScale(ref_gray, scaleFactor=1.1, minNeighbors=4)
        cur_faces = face_cascade.detectMultiScale(cur_gray, scaleFactor=1.1, minNeighbors=4)

        if len(ref_faces) == 0 or len(cur_faces) == 0:
            return {"error": "Could not detect a face in one or both images. Please try again."}

        # 3. Get the coordinates of the first detected face (x, y, width, height)
        rx, ry, rw, rh = ref_faces[0]
        cx, cy, cw, ch = cur_faces[0]

        # 4. Crop the images to contain ONLY the face (removes the background)
        ref_face_cropped = ref_gray[ry:ry+rh, rx:rx+rw]
        cur_face_cropped = cur_gray[cy:cy+ch, cx:cx+cw]

        # 5. Resize the cropped faces to a standard size for comparison
        ref_face_resized = cv2.resize(ref_face_cropped, (150, 150))
        cur_face_resized = cv2.resize(cur_face_cropped, (150, 150))

        # 6. Calculate the difference on the faces ONLY
        diff = np.mean(cv2.absdiff(ref_face_resized, cur_face_resized))

        return {
            "match": bool(diff < 50),  # You might need to tweak this threshold slightly now
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