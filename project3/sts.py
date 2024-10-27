# Sentence Transformers - encode 기반의 task
# https://sbert.net/

# STEP 1 : import module
from sentence_transformers import SentenceTransformer

# STEP 2 : create inference object(instance)
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# STEP 3 : prepare data
sentences1 = "집에 갑시다",
sentences2 = "안녕하세요",

# STEP 4 : inference
embedding1 = model.encode(sentences1)
embedding2 = model.encode(sentences2)
print(embedding1.shape)

# STEP 5 : post processing
similarities = model.similarity(embedding1, embedding2)
print(similarities)
# (1, 384)
# tensor([[0.8385]])
