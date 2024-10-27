# Text classification
# https://huggingface.co/docs/transformers/tasks/sequence_classification

# STEP 1 : import module
from transformers import pipeline

# STEP 2 : create inference object(instance)
classifier = pipeline("sentiment-analysis",
                      model="sangrimlee/bert-base-multilingual-cased-nsmc")

# STEP 3 : prepare data
text = "도대체 뭘 만드신거에요?"

# STEP 4 : inference
result = classifier(text)

# STEP 5 : post processing
print(result)
# [{'label': 'negative', 'score': 0.9698401093482971}]
