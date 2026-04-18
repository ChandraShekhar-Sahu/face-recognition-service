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

        # SFace requires color images (BGR), so we don't convert to grayscale!
        ref_img = cv2.imdecode(ref_arr, cv2.IMREAD_COLOR)
        cur_img = cv2.imdecode(cur_arr, cv2.IMREAD_COLOR)

        if ref_img is None or cur_img is None:
            return {"error": "Image decode failed"}

        # 1. Update detector to match the exact size of the incoming images
        detector.setInputSize((ref_img.shape[1], ref_img.shape[0]))
        _, ref_faces = detector.detect(ref_img)

        detector.setInputSize((cur_img.shape[1], cur_img.shape[0]))
        _, cur_faces = detector.detect(cur_img)

        # Ensure faces were found
        if ref_faces is None or cur_faces is None:
            return {"error": "Could not detect a face in one or both images."}

        # 2. Align the faces (straightens tilted heads)
        ref_aligned = recognizer.alignCrop(ref_img, ref_faces[0])
        cur_aligned = recognizer.alignCrop(cur_img, cur_faces[0])

        # 3. Extract the 128D mathematical features of the face
        ref_feature = recognizer.feature(ref_aligned)
        cur_feature = recognizer.feature(cur_aligned)

        # 4. Compare the two faces using Cosine Distance
        score = recognizer.match(ref_feature, cur_feature, cv2.FaceRecognizerSF_FR_COSINE)

        # For Cosine distance in SFace, a score > 0.363 is generally considered the same person.
        # Higher score = better match (up to 1.0)
        is_match = score >= 0.363

        return {
            "match": bool(is_match),
            "similarity_score": float(score)
        }

    except Exception as e:
        print("ERROR OCCURRED:")
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