# step 1 : import module
from transformers import pipeline


# step 2 : create inference object(instance)
classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_qa_model")


from fastapi import FastAPI, Form

app = FastAPI()


@app.post("/inference/")
async def inference(text: str = Form()):
    # # step 3 : prepare data
    # text = "Tiffany Doll - Wine Makes Me Anal (31.03.2018)_1080p.mp4"


    # step 4 : inferece
    result = classifier(text)

    # step 5 : post processing
    print(result)

    return {"result": result}

'''
text

fast api 튜토리얼 폼데이터로 들어간다

복사 붙여넣기
패스워드 지우고
유저네임을 result로 변경

cls 에서 step1~2 복사 맨 위 상단에 붙여넣기

cls에서 step3~5 복사 inference 함수 안에 붙여넣기

step3 주석처리
- step 3을 주석 처리한 이유는 API 엔드포인트에 전송된 데이터가 text 매개변수로 이미 전달되기 때문입니다.

기존에 step 3의 text 변수에 특정 문자열을 하드코딩한 부분은 실제 서비스 환경에서 유동적으로 입력된 텍스트를 처리하지 못할 수 있으므로, 
이를 주석 처리하고 inference 함수의 매개변수로 전달된 text 값을 그대로 사용하게 됩니다. 이로 인해 매 API 요청 시 클라이언트에서 전달한 텍스트 데이터가 동적으로 처리될 수 있습니다

파이토치 버전이 업그레이드 됨

다운그레이 방법
pip uninstall python-multipart

pip install python-multipart==0.0.12

'''