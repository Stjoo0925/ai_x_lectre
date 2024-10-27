# Multiple choice
# https://huggingface.co/docs/transformers/tasks/multiple_choice

# STEP 1 : import module
from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice
import torch

# STEP 2 : create inference object(instance)
tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_swag_model")
model = AutoModelForMultipleChoice.from_pretrained(
    "stevhliu/my_awesome_swag_model")
labels = torch.tensor(0).unsqueeze(0)

# STEP 3 : prepare data
prompt = "France has a bread law, Le Décret Pain, with strict rules on what is allowed in a traditional baguette."
candidate1 = "The law does not apply to croissants and brioche."
candidate2 = "The law applies to baguettes."

# STEP 4 : inference
inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]],
                   return_tensors="pt", padding=True)
outputs = model(**{k: v.unsqueeze(0)
                for k, v in inputs.items()}, labels=labels)
logits = outputs.logits
predicted_class = logits.argmax().item()
predicted_class

# STEP 5 : post processing
print(predicted_class)
# '0'
