# STEP 1: Import the necessary modules.
import cv2
from fastapi import FastAPI, UploadFile
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(
    model_asset_path='models\efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.112)
detector = vision.ObjectDetector.create_from_options(options)


app = FastAPI()


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()

    # STEP 3: Load the input image.
    binary = np.fromstring(contents, dtype=np.uint8)
    cv_mat = cv2.imdecode(binary, cv2.IMREAD_COLOR)
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)

    # STEP 4: Detect objects in the input image.
    detection_result = detector.detect(rgb_frame)

    # STEP 5: Process the detection result. In this case, visualize it.
    person_count = 0

    for detection in detection_result.detections:
        category = detection.categories[0]
        if category.category_name == 'person':
            person_count += 1

    return {"result": person_count}

# fastapi dev api_det.py

# done

"""
1. 필요한 모듈 가져오기 (STEP 1):
    cv2, fastapi, numpy, mediapipe 등의 모듈을 가져옵니다.
    이 모듈들은 이미지 처리(cv2), HTTP 요청(FastAPI), 수치 계산(numpy), 미디어 데이터 분석(mediapipe) 등에 사용됩니다.

2. 객체 감지기(ObjectDetector) 객체 생성 (STEP 2):
    mediapipe의 ObjectDetector 객체를 생성합니다.
    BaseOptions로 모델 경로(efficientdet_lite0.tflite)를 설정하고, 감지 임계값을 0.112로 설정한 후 ObjectDetector를 생성합니다. 
    이 모델은 이미지 내에서 객체를 감지합니다.

3. FastAPI 애플리케이션 생성:
    FastAPI 애플리케이션 인스턴스를 만듭니다.
    파일 업로드 경로 설정 ("/uploadfile/"):
    /uploadfile/ 경로에 POST 요청을 정의하여 이미지 파일을 업로드할 수 있게 합니다.
    create_upload_file 함수는 UploadFile 객체로 이미지 파일을 받습니다.

4. 입력 이미지 로드 (STEP 3):
    업로드된 파일을 바이트 형태로 읽고, numpy를 사용해 OpenCV 이미지 형식으로 디코딩합니다.
    mediapipe에서 사용할 수 있도록 이미지 데이터를 RGB 포맷으로 변환해 mp.Image 형식으로 준비합니다.

5. 입력 이미지에서 객체 감지 (STEP 4):
    ObjectDetector를 통해 이미지에서 객체를 감지합니다.
    감지 결과는 detection_result 객체에 저장됩니다.

6. 결과 처리 및 시각화 (STEP 5):
    감지된 객체들 중 category_name이 person인 경우를 카운트하여 사람의 수를 셉니다.
    person_count를 최종 결과로 반환합니다.

7. FastAPI 앱 실행:
    마지막 주석 # fastapi dev api_det.py는 이 파일을 FastAPI 서버로 실행하는 명령입니다.
"""
