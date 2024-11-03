import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as ptud
import torch_geometric as ptg
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import os
import time
import random
import json
import sentence_transformers
import openai as oai
import csv
import dotenv
from tqdm.auto import tqdm
import pandas as pd
from numpy.linalg import norm
from sklearn.cluster import KMeans
from fakerFile import (
    generate_fake_response,
    required_questions,
    optional_questions,
    generate_fake_profile,
)
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

mySentenceTransformer = sentence_transformers.SentenceTransformer(
    "paraphrase-MiniLM-L6-v2"
)
dotenv.load_dotenv()
oai.api_key = os.getenv("OPENAI_API_KEY")

possiblePeople = []

for i in tqdm(range(150)):
    fakerResponses = generate_fake_profile()
    possiblePeople.append(fakerResponses)

with open("possiblePeople2.json", "w") as file:
    json.dump(possiblePeople, file, indent=4)

# #possible people to json
with open("possiblePeople2.json", "w") as file:
    json.dump(possiblePeople, file)


#     response = oai.ChatCompletion.create(
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": "Be a good assistant"
#              + "I want you to create a random college student and answer the questions given. Please do vary the occupations do not make everyone a part time barista please majors cleanliness levels, and everything else so I can have good data. Also please seperate the answers with a comma and no apostrophes inside and make it one line"},
#             {"role": "user", "content": " What is your age? What is your education level? What is your occupation? What is your major? What is your level of cleanliness? What is your level of noise tolerance? What time do you usually go to bed? What time do you usually wake up? Do you smoke? Do you drink? Do you have any pets? Do you have any dietary restrictions? What is your preferred number of roommates?"
#              + "What is your budget? What is your preferred move-in date? What is your preferred lease length? What is your preferred neighborhood?"}
#         ]
#     )

#     possiblePeople.append(response['choices'][0]['message']['content'])


# #possible people to json
# with open('possiblePeople.json', 'w') as file:
#     json.dump(possiblePeople, file)

# GNN to node classify people from possible people


# # Load possible people from json
with open("possiblePeople2.json", "r") as file:
    possiblePeople = json.load(file)
embeddings = mySentenceTransformer.encode(possiblePeople)
print(embeddings)
edgelist1 = []
edgelist2 = []
minin = 10
sum = 0
for i in range(150):
    for j in range(150):
        if (
            i != j
            and np.dot(embeddings[i], embeddings[j])
            / norm(embeddings[i])
            / norm(embeddings[j])
            > 0.7
        ):
            edgelist1.append(i)
            edgelist2.append(j)
with open("possiblePeople2.json", "r") as file:
    possiblePeople = json.load(file)
print(sum / (50 * 49))
print(minin)


x = torch.tensor(embeddings, dtype=torch.float)


# Create a simple graph with 4 nodes and 4 edges
edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)


edge_index = torch.tensor([edgelist1, edgelist2], dtype=torch.long)
# x = torch.tensor([[1], [2], [3], [4]], dtype=torch.float)  # Node features

# Create a Data object
data = Data(x=x, edge_index=edge_index)


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(384, 48)  # Input: 1 feature, Output: 2 features
        self.conv2 = GCNConv(48, 384)  # Input: 2 features, Output: 2 features

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(200):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    out = model(data)

    # Apply a simple reconstruction loss (could be any unsupervised loss)
    loss = F.mse_loss(
        out, data.x
    )  # You may want to use a different method for your embeddings
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


# Get the node embeddings
model.eval()
embeddings = model(data).detach().numpy()

# Perform K-Means clustering
kmeans = KMeans(n_clusters=12)  # Choose number of clusters
predicted_labels = kmeans.fit_predict(embeddings)

print("Predicted Clusters:", predicted_labels)
for i in range(12):
    print(
        "Cluster",
        i,
        ":",
        [possiblePeople[j] for j in range(50) if predicted_labels[j] == i],
    )


df = pd.DataFrame(possiblePeople)
print(df.columns)


df["What is your budget?"] = (
    df["What is your budget?"]
    .str.replace("$", "")
    .str.replace(" per month", "")
    .astype(float)
)
df["What time do you usually go to bed?"] = (
    df["What time do you usually go to bed?"].str.replace(" PM", "").astype(int)
)
# df['What is your preferred lease length?'] = df['What is your preferred lease length?'].str.replace(' months', '').astype(int)


def convert_lease_length(length):
    length = length.lower()
    if length == "1 year":
        return 12
    out = int(length.split()[0])
    return out


# Apply the function to the 'What is your preferred lease length?' column
# df['What is your preferred lease length'] = df['What is your preferred lease length?'].apply(convert_lease_length)
# df.drop('What is your preferred lease length?', axis=1, inplace=True)
# Normalize features
features = df[
    ["What is your age?", "What is your budget?", "What time do you usually go to bed?"]
].values
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Construct similarity matrix
similarity_matrix = cosine_similarity(features_normalized)
threshold = 0.5  # Adjust this threshold based on your needs
adjacency_matrix = (similarity_matrix > threshold).astype(int)
# Create a graph from the similarity matrix
G = nx.from_numpy_array(adjacency_matrix)

# Calculate centrality (degree centrality as an example)
centrality = nx.degree_centrality(G)
# Convert centrality to DataFrame for easy handling
centrality_df = pd.DataFrame(list(centrality.items()), columns=["Node", "Centrality"])
centrality_df["Profile"] = df["What is your name?"]

# Find outliers: Here we define outliers as those with lower centrality
threshold = np.percentile(
    centrality_df["Centrality"], 5
)  # You can adjust the threshold

outliers = centrality_df[centrality_df["Centrality"] < threshold]

# Output outliers
print("Identified Outliers:")
print(outliers[["Profile", "Centrality"]])
