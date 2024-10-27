# Question answering
# https://huggingface.co/docs/transformers/tasks/question_answering

# STEP 1 : import module
from transformers import pipeline

# STEP 2 : create inference object(instance)
question_answerer = pipeline(
    "question-answering", model="stevhliu/my_awesome_billsum_model")

# STEP 3 : prepare data
question = "How many programming languages does BLOOM support?"
context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."

# STEP 4 : inference
result = question_answerer(question=question, context=context)

# STEP 5 : post processing
print(result)
# {'score': 0.004685470834374428, 'start': 0, 'end': 9, 'answer': 'BLOOM has'}
