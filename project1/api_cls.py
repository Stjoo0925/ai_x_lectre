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
