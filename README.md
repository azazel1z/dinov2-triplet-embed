# DINOv2 Triplet Embedding

*An end-to-end pipeline for image embedding generation using Meta's DINOv2, optimized with Triplet Loss and integrated with a vector database for semantic similarity search.*

## Overview

This repository is a portfolio project demonstrating how to build a visual similarity search engine. It relies on the visual features of DINOv2, fine-tuned with a Triplet Loss objective, to generate distinct image embeddings. These vectors are then indexed in a vector database for nearest-neighbor image retrieval.

> **Note:** This repository is intended to showcase a machine learning workflow, methodology, and architecture, rather than serving as an out-of-the-box library for production use. 

## Methodology & Workflow

### 1. Model Fine-Tuning (`train/`)
This project adapts DINOv2 for metric learning using a Triplet Margin Loss setup:
- **Anchor:** The baseline image.
- **Positive:** An image matching the anchor's class.
- **Negative:** An image from a different class.

This objective forces the model to group similar images closer together in the latent space while pushing dissimilar images apart.

### 2. Feature Vectorization (`vectorizeDINO.ipynb`)
Once the model is fine-tuned, this notebook processes the target dataset through the adapted DINOv2 backbone. The images are converted into dense, high-dimensional vector embeddings.

### 3. Vector Database Integration (`vector_upsert.ipynb`)
To enable retrieval, the generated embeddings need to be indexed. This notebook formats the extracted PyTorch tensors and upserts them into a vector database, allowing for standard vector similarity search (e.g., querying the database for images visually similar to a specific input).