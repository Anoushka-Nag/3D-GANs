from sentence_transformers import SentenceTransformer, util

model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

sentences = [
    "This is a sentence about natural language processing.",
    "This sentence focuses on generating text embeddings.",
    "Machine learning is a powerful tool for various applications."
]
sentence_embeddings = model.encode(sentences)

query = "What are embeddings used for?"
query_embedding = model.encode(query)

for sentence, embedding in zip(sentences, sentence_embeddings):
    similarity = util.cos_sim(query_embedding, embedding)
    print(f"Similarity between query and '{sentence}' is {similarity.item()}")
