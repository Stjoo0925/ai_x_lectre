# easyocr

# pip install "fastapi[standard]"


# STEP 1
from fastapi import FastAPI, UploadFile
import easyocr
import cv2
import numpy as np

# STEP 2: Create an ImageClassifier object.
reader = easyocr.Reader(['ch_sim','en'])

# FastAPI
app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    contents = await file.read()

    # STEP 3: Load the input image.
    binary = np.fromstring(contents, dtype=np.uint8)
    cv_mat = cv2.imdecode(binary, cv2.IMREAD_COLOR)

    # STEP 4: Classify the input image.
    result = reader.readtext(cv_mat)
    print(result)

    # STEP 5: Process the classification result. In this case, visualize it.
    return {"result": result.item()}

# fastapi dev api_ocr.py