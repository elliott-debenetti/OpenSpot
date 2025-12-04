import os
import io
import time
import requests
import zipfile
import torch
import numpy as np
import matplotlib.pyplot as plt

os.chdir('/Users/elliott/Documents/GitHub/OpenSpot')
print("You are now in:", os.getcwd())

from dataset import acpds
from utils import transforms
from utils import visualize as vis

train_ds, valid_ds, test_ds = acpds.create_datasets('dataset/data')

image_batch, rois_batch, labels_batch = next(iter(valid_ds))
image_raw, rois, labels = image_batch[0], rois_batch[0], labels_batch[0]
image = transforms.preprocess(image_raw, res=1080)
vis.plot_ds_image(image, rois, labels, show=True)

# create model
from models.rcnn import RCNN
model = RCNN()

# load model weights
weights_path = '/Users/elliott/Documents/GitHub/parking-space-occupancy/out_dir/RCNN_v2/weights_last_epoch.pt'
model.load_state_dict(torch.load(weights_path, map_location='cpu'))

# # plot test predictions
# for i, (image_batch, rois_batch, labels_batch) in enumerate(test_ds):
#     if i == 2: break
#     image, rois, labels = image_batch[0], rois_batch[0], labels_batch[0]
#     image = transforms.preprocess(image)
#     with torch.no_grad():
#         class_logits = model(image, rois)
#         class_scores = class_logits.softmax(1)[:, 1]
#     vis.plot_ds_image(image, rois, class_scores, labels)    

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm

# ================================
# 1. EVALUATE PERFORMANCE ON TEST SET
# ================================
y_true = []
y_pred = []

model.eval()

print("\nEvaluating test set performance...")

for image_batch, rois_batch, labels_batch in tqdm(test_ds):
    image = transforms.preprocess(image_batch[0])
    rois = rois_batch[0]
    labels = labels_batch[0].numpy()

    with torch.no_grad():
        logits = model(image, rois)
        preds = logits.softmax(1)[:, 1].numpy()
        preds_binary = (preds >= 0.5).astype(int)

    y_true.extend(labels.tolist())
    y_pred.extend(preds_binary.tolist())

# Compute global test metrics
test_accuracy = accuracy_score(y_true, y_pred)
test_precision = precision_score(y_true, y_pred)
test_recall = recall_score(y_true, y_pred)
test_f1 = f1_score(y_true, y_pred)

print("\n=== TEST SET PERFORMANCE ===")
print(f"Accuracy : {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall   : {test_recall:.4f}")
print(f"F1-score : {test_f1:.4f}")

# ================================
# 2. COMPUTE PER-ROI METRICS ON TRAINING SET
# ================================
roi_records = []

print("\nComputing per-ROI performance on testing set...")

for image_batch, rois_batch, labels_batch in tqdm(test_ds):
    image = transforms.preprocess(image_batch[0])
    rois = rois_batch[0]
    labels = labels_batch[0].numpy()

    with torch.no_grad():
        logits = model(image, rois)
        preds = logits.softmax(1)[:, 1].numpy()
        preds_binary = (preds >= 0.5).astype(int)

    # Loop through ROIs individually
    for i, roi in enumerate(rois):
        roi_records.append({
            "roi_index": i,
            "true_label": int(labels[i]),
            "pred_label": int(preds_binary[i]),
            "probability": float(preds[i])
        })

# Convert to DataFrame
df = pd.DataFrame(roi_records)

# Compute metrics per ROI index
roi_metrics = df.groupby("roi_index").apply(
    lambda x: pd.Series({
        "precision": precision_score(x["true_label"], x["pred_label"]),
        "recall": recall_score(x["true_label"], x["pred_label"]),
        "f1_score": f1_score(x["true_label"], x["pred_label"])
    })
)

# Save CSV
output_csv = "roi_performance_metrics.csv"
roi_metrics.to_csv(output_csv)
print(f"\nROI metrics saved to: {output_csv}")

