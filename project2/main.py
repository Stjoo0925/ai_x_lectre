# -- 환경설정 --

# conda create -n project2 python=3.12

# conda activate project2

# 얼굴 인식 오픈소스 Repo - insightface
# https://github.com/deepinsight/insightface

# 오픈소스 패키지
# https://pypi.org/project/insightface/

# 설치 명령어 
# pip install insightface

# error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools"
# https://visualstudio.microsoft.com/ko/visual-cpp-build-tools/
# C++을 이용한 데스크톱 개발 체크
# 설치후 다시 pip install insightface

# Successfully built insightface - 완료

# -- 개념설명 --

# 임베딩 [Embedding]
    # 데이터 입력 > 모델로 처리 > 숫자의 집합으로 변환
    # 변환된 숫자들은 고유한 차원의 개념을 가짐
    # 따라서 각 데이터들의 유사도를 바탕으로 각종 로직을 구성할수 있음
    # https://platform.openai.com/docs/models/embeddings - 임베딩 예제