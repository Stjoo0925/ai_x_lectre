# FastAPI
# https://fastapi.tiangolo.com/

# pip install "fastapi[standard]"

# STEP 1: Import modules.
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object.
base_options = python.BaseOptions(
    model_asset_path='models\efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=3)
classifier = vision.ImageClassifier.create_from_options(options)

# FastAPI
app = FastAPI()


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()

    # contents <- burger.jpg(bytes)

    # STEP 3: Load the input image.
    binary = np.fromstring(contents, dtype=np.uint8)
    cv_mat = cv2.imdecode(binary, cv2.IMREAD_COLOR)

    # cv_mat = cv2.imread(contents)
    # cv2.imread
    # file = file.open('path')
    # cv_mat = cv2.imdecode(file)

    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)

    # 흑백이미지 파일
    # gray_frame = mp.Image(
    #     image_format=mp.ImageFormat.GRAY8,
    #     data=cv2.cvtColor(cv_mat, cv2.COLOR_RGB2GRAY))

    # image = mp.Image.create_from_file('image\\burger.jpg')

    # mp.Image.create_from_file : burger.jpg -> decoding(bitmap) -> convert to tensor

    # STEP 4: Classify the input image.
    classification_result = classifier.classify(rgb_frame)
    # print(classification_result)

    # STEP 5: Process the classification result. In this case, visualize it.
    top_category = classification_result.classifications[0].categories[0]
    print(f"{top_category.category_name} ({top_category.score:.2f})")

    return {"category": top_category.category_name,
            "score": f"{top_category.score:.2f}"}

# fastapi dev api_cls.py

# done

"""
# Api Description

1. 모듈 가져오기 (STEP 1): 필요한 라이브러리와 모듈을 가져옵니다. 
    여기에는 FastAPI, numpy, opencv-python(cv2), mediapipe, 
    그리고 mediapipe에서 이미지 분류를 위한 컴포넌트가 포함됩니다.

2. 분류기 객체 생성 (STEP 2): ImageClassifier 객체를 생성합니다. 
    mediapipe의 ImageClassifierOptions을 설정해 분류 모델을 로드하고, 모델의 결과를 3개로 제한합니다. 
    efficientnet_lite0.tflite 모델을 사용하도록 지정합니다.

3. FastAPI 애플리케이션 생성: FastAPI 애플리케이션 인스턴스를 만듭니다.

4. 파일 업로드 경로 설정: /uploadfile/ 경로에 POST 요청을 정의하여 이미지 파일을 업로드할 수 있도록 합니다.

5. 이미지 로드 및 디코딩 (STEP 3): UploadFile로 받은 이미지를 cv2와 numpy를 이용해 바이트에서 OpenCV 이미지 형식으로 디코딩합니다.

6. 이미지 분류 (STEP 4): mediapipe의 ImageClassifier를 사용해 이미지를 분류합니다. 
    이 단계에서 모델이 이미지의 카테고리를 예측합니다.

7. 결과 반환 (STEP 5): 분류 결과 중 가장 높은 카테고리와 점수를 추출하여 API 응답으로 반환합니다.
"""

"""
# Code Organization

# STEP 1: Import modules.
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object.
base_options = python.BaseOptions(
    model_asset_path='models\efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=3)
classifier = vision.ImageClassifier.create_from_options(options)

# FastAPI
app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()

    # STEP 3: Load the input image.
    binary = np.fromstring(contents, dtype=np.uint8)
    cv_mat = cv2.imdecode(binary, cv2.IMREAD_COLOR)
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_mat)

    # STEP 4: Classify the input image.
    classification_result = classifier.classify(rgb_frame)

    # STEP 5: Process the classification result. In this case, visualize it.
    top_category = classification_result.classifications[0].categories[0]
    print(f"{top_category.category_name} ({top_category.score:.2f})")

    return {"category": top_category.category_name,
            "score": f"{top_category.score:.2f}"}

# $ fastapi dev api_cls.py
"""
