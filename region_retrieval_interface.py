import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageFont
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pickle
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms as T
from models.vit_lora import vit_lora

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vit_lora(ckpt_path='checkpoints\\ibot_horae_lora_7e-3_30.pth').to(device)
transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

# Catalog
catalog_path = Path("D:\\Universidad\\Proyecto de Titulo\\sam-dino_detector\\lora_output")
images_path = Path("D:\\Universidad\\Proyecto de Titulo\\sam-dino_detector\\flat_textures")
with open(catalog_path / "all_features_ibot_horae_lora_7e-3_30.pkl", "rb") as f:
    catalog = pickle.load(f)

region_data = []
for image_id, image_data in catalog["processed_images"].items():
    if "regions" in image_data:
        for region in image_data["regions"]:
            clean_image_id = image_id.replace('_flat.png', '')
            region_data.append({
                "id": f"{clean_image_id}_{region['region_id']}",
                "image_path": f"{catalog_path}\\{clean_image_id}_flat\\thumbnails\\region_{region['region_id']:04d}.png",
                "features": region["features"],
                "bbox": region["bbox"]
            })  

# Similarity calculation
def compute_similarity(features1, features2):
    """Compute similarity score between two feature vectors."""
    query_norm = normalize(features1.reshape(1, -1), norm='l2')
    target_norm = normalize(features2.reshape(1, -1), norm='l2')
    sim = cosine_similarity(query_norm, target_norm)[0][0]
    return sim

# Get top-N similar regions
def get_most_similar(query_features, top_n=30):
    similarities = []
    for region in region_data:
        sim = compute_similarity(query_features, region["features"])
        similarities.append((region, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

# Draw bounding boxes
def draw_bboxes(image_path, regions_with_scores, margin=20):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    img = img.crop((0, 0, width // 2, height))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    
    for region, sim in regions_with_scores:
        x, y, w, h = region["bbox"]
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(img.width, x + w + margin), min(img.height, y + h + margin)
        
        draw.rectangle([x0, y0, x1, y1], outline="#00FF00", width=2)
        draw.text((x0, max(0, y0 - 10)), f"{sim:.2f}", fill="#FFFFFF", font=font)

    
    
    return img

# Extract features from external image
def extract_features_from_image(image):
    image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(image_tensor)
    return features.cpu().numpy()

# Load external image
def load_external_image():
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
    )
    if not file_path:
        return
    img_q = Image.open(file_path)
    features = extract_features_from_image(img_q)

    top_regions = get_most_similar(features, top_n=30)
    most_similar_region, top_score = top_regions[0]
    
    img_q.thumbnail((200, 200))
    img_q_tk = ImageTk.PhotoImage(img_q)
    query_img_label.config(image=img_q_tk)
    query_img_label.image = img_q_tk
    query_img_label_text.set("External Query")
    
    img_m = Image.open(most_similar_region["image_path"])
    img_m.thumbnail((200, 200))
    img_m_tk = ImageTk.PhotoImage(img_m)
    similar_img_label.config(image=img_m_tk)
    similar_img_label.image = img_m_tk
    similar_img_label_text.set(f"Most Similar ({top_score:.2f})")
    
    same_image_regions = [
        (reg, score) for reg, score in top_regions
        if reg["id"].split("_")[0] == most_similar_region["id"].split("_")[0]
    ]
    source_image_id = most_similar_region['id'].split("_")[0]
    source_img_path = f"{images_path}\\{source_image_id}_flat.png"

    annotated = draw_bboxes(source_img_path, same_image_regions)
    annotated.save(f"annotated_output_{most_similar_region['id']}.png")
    annotated.thumbnail((1500, 750))
    annotated_tk = ImageTk.PhotoImage(annotated)
    annotated_img_label.config(image=annotated_tk)
    annotated_img_label.image = annotated_tk
    annotated_img_label_text.set("Annotated Same-Image Regions")

# Update display
def on_query_select(event=None):
    query_idx = query_dropdown.current()
    if query_idx < 0 or query_idx >= len(region_data):
        return

    query_region = region_data[query_idx]
    top_regions = get_most_similar(query_region["features"], top_n=30)
    most_similar_region, top_score = top_regions[0]
    
    img_q = Image.open(query_region["image_path"])
    img_q.thumbnail((200, 200))
    img_q_tk = ImageTk.PhotoImage(img_q)
    query_img_label.config(image=img_q_tk)
    query_img_label.image = img_q_tk
    query_img_label_text.set(f"Query Image ({query_region['id']})")
    
    img_m = Image.open(most_similar_region["image_path"])
    img_m.thumbnail((200, 200))
    img_m_tk = ImageTk.PhotoImage(img_m)
    similar_img_label.config(image=img_m_tk)
    similar_img_label.image = img_m_tk
    similar_img_label_text.set(f"Most Similar ({top_score:.2f})")
    
    same_image_regions = [
        (reg, score) for reg, score in top_regions
        if reg["id"].split("_")[0] == most_similar_region["id"].split("_")[0]
    ]
    source_image_id = most_similar_region['id'].split("_")[0]
    source_img_path = f"{images_path}\\{source_image_id}_flat.png"

    annotated = draw_bboxes(source_img_path, same_image_regions)
    annotated.thumbnail((2000, 1000))
    annotated_tk = ImageTk.PhotoImage(annotated)
    annotated_img_label.config(image=annotated_tk)
    annotated_img_label.image = annotated_tk
    annotated_img_label_text.set("Annotated Same-Image Regions")

# Tkinter UI setup
root = tk.Tk()
root.title("Region Similarity Search")

# Dropdown for query selection
query_options = [f"{i['id']}" for i in region_data]
query_dropdown = ttk.Combobox(root, values=query_options, width=60)
query_dropdown.grid(row=0, column=0, pady=5)
query_dropdown.bind("<<ComboboxSelected>>", on_query_select)

# Button to load external image
load_button = tk.Button(root, text="Select External Image", command=load_external_image)
load_button.grid(row=0, column=1, padx=5, pady=5)

# Labels for images
query_img_label_text = tk.StringVar()
similar_img_label_text = tk.StringVar()
annotated_img_label_text = tk.StringVar()

tk.Label(root, textvariable=query_img_label_text).grid(row=1, column=0)
tk.Label(root, textvariable=similar_img_label_text).grid(row=1, column=1)
tk.Label(root, textvariable=annotated_img_label_text).grid(row=3, column=0, columnspan=2)

query_img_label = tk.Label(root)
similar_img_label = tk.Label(root)
annotated_img_label = tk.Label(root)
query_img_label.grid(row=2, column=0, padx=5, pady=5)
similar_img_label.grid(row=2, column=1, padx=5, pady=5)
annotated_img_label.grid(row=4, column=0, padx=5, pady=5, columnspan=2)

root.mainloop()