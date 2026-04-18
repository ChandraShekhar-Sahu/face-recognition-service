import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import traceback
import time
import mediapipe as mp

app = FastAPI()

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------- 1. INITIALIZE SETTINGS & AI MODELS ---------------------- #

UPLOAD_DIR = "api/uploads/"
REFERENCE_IMAGE_PATH = os.path.join(UPLOAD_DIR, "reference_face.jpg")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 🔥 1. Get the directory where this Python script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔥 2. Point explicitly to the 'models' folder inside that directory
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 🔥 3. Create the full paths to the ONNX files inside the models folder
DETECTOR_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
RECOGNIZER_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

# Initialize SFace using the new correct paths
detector = cv2.FaceDetectorYN.create(DETECTOR_PATH, "", (300, 300))
recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_PATH, "")

# MediaPipe (For Liveness & Head Tracking)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

liveness_state = {"last_blink_time": time.time(), "blink_detected": False}


# ---------------------- 2. HELPER FUNCTIONS ---------------------- #

def calculate_ear(eye_points):
    v1 = np.linalg.norm(eye_points[1] - eye_points[5])
    v2 = np.linalg.norm(eye_points[2] - eye_points[4])
    h = np.linalg.norm(eye_points[0] - eye_points[3])
    return (v1 + v2) / (2.0 * h)

def get_head_pose(face_landmarks, img_w, img_h):
    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
    ])
    indices = [1, 152, 226, 446, 57, 287] 
    image_points = np.array([
        (face_landmarks.landmark[i].x * img_w, face_landmarks.landmark[i].y * img_h) 
        for i in indices
    ], dtype="double")

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    
    success, rotation_vector, _ = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    if success:
        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        yaw = angles[1]
        if yaw > 15: return "Looking Right"
        elif yaw < -15: return "Looking Left"
        else: return "Looking Center"
    return "Unknown"


# ---------------------- 3. ENDPOINTS (MATCHING DJANGO EXACTLY) ---------------------- #

@app.post("/api/upload_photo")
async def upload_photo(image: UploadFile = File(...)):
    """Mimics the Django upload_photo logic"""
    try:
        contents = await image.read()
        with open(REFERENCE_IMAGE_PATH, "wb") as f:
            f.write(contents)
        return JSONResponse({"message": "Reference photo uploaded successfully!", "path": REFERENCE_IMAGE_PATH})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/api/verify_face")
async def verify_face(image: UploadFile = File(...)):
    """Mimics the Django verify_face logic and returns the exact same JSON"""
    try:
        # 1. Read the uploaded webcam image
        contents = await image.read()
        np_arr = np.frombuffer(contents, np.uint8)
        cur_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if cur_img is None:
            return JSONResponse({"error": "Invalid image file"}, status_code=400)

        # 2. Check if reference image exists
        if not os.path.exists(REFERENCE_IMAGE_PATH):
            return JSONResponse({"error": "No face found in reference image"}, status_code=400)
        
        ref_img = cv2.imread(REFERENCE_IMAGE_PATH)

        # 3. Detect faces using SFace
        detector.setInputSize((ref_img.shape[1], ref_img.shape[0]))
        _, ref_faces = detector.detect(ref_img)
        
        detector.setInputSize((cur_img.shape[1], cur_img.shape[0]))
        _, cur_faces = detector.detect(cur_img)

        if ref_faces is None or cur_faces is None:
            return JSONResponse({
                "match": False,
                "face_detected": False, 
                "message": "No face detected in uploaded image"
            })

        # 4. Compare Faces (SFace logic)
        ref_aligned = recognizer.alignCrop(ref_img, ref_faces[0])
        cur_aligned = recognizer.alignCrop(cur_img, cur_faces[0])
        ref_feature = recognizer.feature(ref_aligned)
        cur_feature = recognizer.feature(cur_aligned)
        
        score = recognizer.match(ref_feature, cur_feature, cv2.FaceRecognizerSF_FR_COSINE)
        match = bool(score >= 0.363)
        message = "Face Matched!" if match else "Face Not Matched!"

        # 5. Liveness and Head Movement (MediaPipe)
        img_h, img_w, _ = cur_img.shape
        rgb_frame = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return JSONResponse({
                "match": match,
                "face_detected": False,
                "message": message,
                "liveness": False,
                "liveness_text": "No Face Detected",
                "head_movement": "unknown"
            })

        landmarks = results.multi_face_landmarks[0]
        
        # Head Movement Logic
        raw_head_status = get_head_pose(landmarks, img_w, img_h)
        if raw_head_status in ["Looking Left", "Looking Right"]:
            head_movement_formatted = "suspicious"
        elif raw_head_status == "Looking Center":
            head_movement_formatted = "normal"
        else:
            head_movement_formatted = "unknown"

        # Liveness (Blink) Logic
        EAR_THRESHOLD = 0.20
        BLINK_INTERVAL = 3.0
        
        left_eye_idx = [33, 160, 158, 133, 153, 144]
        right_eye_idx = [362, 385, 387, 263, 373, 380]
        left_eye_points = np.array([[landmarks.landmark[i].x * img_w, landmarks.landmark[i].y * img_h] for i in left_eye_idx])
        right_eye_points = np.array([[landmarks.landmark[i].x * img_w, landmarks.landmark[i].y * img_h] for i in right_eye_idx])

        ear = (calculate_ear(left_eye_points) + calculate_ear(right_eye_points)) / 2.0
        current_time = time.time()

        if ear < EAR_THRESHOLD:
            if not liveness_state["blink_detected"]:
                liveness_state["blink_detected"] = True
                liveness_state["last_blink_time"] = current_time
        else:
            if current_time - liveness_state["last_blink_time"] > BLINK_INTERVAL:
                liveness_state["blink_detected"] = False

        is_live = current_time - liveness_state["last_blink_time"] <= BLINK_INTERVAL
        liveness_text = "Blink Detected - Real Person" if is_live else "No Recent Blink - Potential Video"

        # 6. EXACT DJANGO JSON RESPONSE
        return JSONResponse({
            "match": match,
            "face_detected": True,
            "message": message,
            "liveness": is_live,
            "liveness_text": liveness_text,
            "head_movement": head_movement_formatted
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=400)