import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# ---- Load data ----
with open("D:\\Universidad\\Proyecto de Titulo\\sam-dino_detector\\lora_output\\all_features_ibot_horae_lora_7e-3_30.pkl", "rb") as f:
    data = pickle.load(f)

# Extract features into a matrix
features = []
for image_id, image_data in data["processed_images"].items():
    if "regions" in image_data:
        for region in image_data["regions"]:
            features.append(region["features"])

# ---- Compute similarities ----
similarity_matrix = cosine_similarity(features)
triu_indices = np.triu_indices_from(similarity_matrix, k=1)
similarities = similarity_matrix[triu_indices]

# ---- Histogram ----
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(similarities, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
plt.title("Similarity Histogram")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.6)

# ---- CDF ----
plt.subplot(1,2,2)
sorted_sim = np.sort(similarities)
cdf = np.arange(1, len(sorted_sim)+1) / len(sorted_sim)

plt.plot(sorted_sim, cdf, color="darkorange", lw=2)
plt.title("Similarity CDF")
plt.xlabel("Cosine Similarity")
plt.ylabel("Cumulative Probability")
plt.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

# ---- Stats ----
print("Number of region pairs:", len(similarities))
print("Mean similarity:", np.mean(similarities))
print("Median similarity:", np.median(similarities))
print("90th percentile similarity:", np.percentile(similarities, 90))
print("95th percentile similarity:", np.percentile(similarities, 95))