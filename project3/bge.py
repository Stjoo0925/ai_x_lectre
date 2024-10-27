# BAAI/bge-m3
# https://huggingface.co/BAAI/bge-m3

# STEP 1 : import module
from FlagEmbedding import BGEM3FlagModel

# STEP 2 : create inference object(instance)
model = BGEM3FlagModel('BAAI/bge-m3',
                       use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight performance degradation

# STEP 3 : prepare data
sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

# STEP 4 : inference
embeddings_1 = model.encode(sentences_1,
                            batch_size=12,
                            # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                            max_length=8192,
                            )['dense_vecs']
embeddings_2 = model.encode(sentences_2)['dense_vecs']
similarity = embeddings_1 @ embeddings_2.T

# STEP 5 : post processing
print(similarity)
# [[0.6265, 0.3477], [0.3499, 0.678 ]]