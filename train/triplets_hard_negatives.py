import streamlit as st
from PIL import Image
import pinecone as pc
from pinecone import Pinecone
import os
import json
from dotenv import load_dotenv
 
load_dotenv()
 
# --- Config ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
IMAGE_DIR = "../data/Designs_bw"
TRIPLET_FILE = "triplet_bw.json"
 
# --- Connect to Pinecone ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("dinov2base")
 
# --- Helper: Extract folder name from image ID ---
def get_folder_from_path(path):
    return os.path.basename(os.path.dirname(path))
 
# --- Load image from disk ---
def load_image(image_id):
    image_path = os.path.normpath(image_id)
    if os.path.exists(image_path):
        return Image.open(image_path)
    full_path = os.path.join(IMAGE_DIR, image_path)
    if not os.path.exists(full_path):
        st.warning(f"Image not found: {full_path}")
        return None
    return Image.open(full_path)
 
# --- Load all image IDs and organize by folder ---
all_ids = []
for id_batch in index.list(namespace="vectors_before"):
    all_ids.extend(id_batch)
all_ids = sorted(all_ids)
 
folder_to_ids = {}
for img_id in all_ids:
    folder = get_folder_from_path(img_id)
    folder_to_ids.setdefault(folder, []).append(img_id)
 
# --- Streamlit UI ---
st.title("Triplet Builder - Visual Query Explorer")
 
# --- Anchor state management ---
if "anchor_index" not in st.session_state:
    st.session_state.anchor_index = 0
 
# --- Show current anchor index (debug/info) ---
st.markdown(f"**Current Anchor Index:** {st.session_state.anchor_index} / {len(all_ids) - 1}")
 
# Optional manual override slider
manual_index = st.slider("Manual Override Anchor Index", 0, len(all_ids) - 1, st.session_state.anchor_index)
if manual_index != st.session_state.anchor_index:
    st.session_state.anchor_index = manual_index
    st.rerun()
 
anchor_index = st.session_state.anchor_index
anchor_id = all_ids[anchor_index]
anchor_img = load_image(anchor_id)
anchor_folder = get_folder_from_path(anchor_id)
 
# --- Show anchor image ---
st.subheader("Anchor Image")
st.image(anchor_img, caption=anchor_id, use_container_width=True)
 
# --- Show all positives (same folder) ---
st.subheader("All Positives from Same Folder")
positives = [pid for pid in folder_to_ids.get(anchor_folder, []) if pid != anchor_id]
 
cols_pos = st.columns(5)
for i, pos_id in enumerate(positives):
    with cols_pos[i % 5]:
        pos_img = load_image(pos_id)
        if pos_img:
            st.image(pos_img, caption=pos_id, use_container_width=True)
 
# --- Query Pinecone ---
result = index.query(id=anchor_id, top_k=20, include_metadata=False, namespace="vectors_before")
 
# --- Select negatives from results with scores (sorted by descending score) ---
st.subheader("Select Negatives (with Similarity Scores)")
selected_negatives = []
cols = st.columns(5)
 
# Sort matches by score descending
sorted_matches = sorted(result['matches'], key=lambda x: x.get('score', 0), reverse=True)
 
for i, match in enumerate(sorted_matches):
    match_id = match['id']
    if match_id == anchor_id:
        continue
    score = match.get('score', 0.0)
    match_img = load_image(match_id)
    with cols[i % 5]:
        if match_img:
            st.image(match_img, caption=f"{match_id}\nScore: {score:.4f}", use_container_width=True)
        if st.checkbox(label=f"Select as Negative", key=f"neg-{match_id}"):
            selected_negatives.append(match_id)
 
# --- Save triplets and move to next anchor ---
if st.button("Save Triplets"):
    triplets = []
    for pos_id in positives:
        for neg_id in selected_negatives:
            triplets.append({
                "anchor": anchor_id,
                "positive": pos_id,
                "negative": neg_id
            })
 
    # Save all triplets into one JSON file
    os.makedirs("triplets", exist_ok=True)
    triplet_path = os.path.join("triplets", TRIPLET_FILE)
 
    # Load existing triplets if file exists
    if os.path.exists(triplet_path):
        with open(triplet_path, "r") as f:
            existing_triplets = json.load(f)
    else:
        existing_triplets = []
 
    existing_triplets.extend(triplets)
 
    with open(triplet_path, "w") as f:
        json.dump(existing_triplets, f, indent=2)
 
    st.success(f"✅ Saved {len(triplets)} triplets to {TRIPLET_FILE}")
 
    # Move to next anchor if not last
    if st.session_state.anchor_index < len(all_ids) - 1:
        st.session_state.anchor_index += 1
        st.rerun()
    else:
        st.info("🎉 All anchors processed!")