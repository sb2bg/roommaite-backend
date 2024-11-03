from sentence_transformers import SentenceTransformer

# Load a pre-trained sentence transformer model. This model's output vectors are of size 384
model = SentenceTransformer("all-MiniLM-L6-v2")
print("model loaded")

prefs = (
    "1. What do you like to do in your free time? I like to read books; 2. I like to "
)
print(model.encode(prefs, normalize_embeddings=True).tolist())
