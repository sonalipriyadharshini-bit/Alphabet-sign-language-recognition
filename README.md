# Alphabet-sign-language-recognition
# ASL Alphabet Recognition (CNN + Transfer Learning)

A deep learning model that recognizes American Sign Language (ASL) alphabet hand signs from images, trained on the [ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (Kaggle) and evaluated with standard classification metrics.

## Overview

- **Task**: 29-class image classification — A–Z, plus `space`, `del`, and `nothing`
- **Approach**: Transfer learning using MobileNetV2 (pretrained on ImageNet), fine-tuned in two phases on the ASL Alphabet dataset
- **Environment**: Google Colab (T4 GPU)
- **Inference**: Upload any hand-sign image and the model predicts the letter, with confidence scores

## Dataset

[ASL Alphabet — Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

- ~87,000 images across 29 classes (~3,000 images/class in the full dataset)
- Since the official test set only contains 1 image per class, the notebook creates its own **stratified 70/15/15 train/validation/test split** from the training data for meaningful evaluation.
- `del`, `space`, and `nothing` are control gestures (not letters) included so the model can also recognize "delete last letter," "insert a space," and "no hand present" — useful for a real fingerspelling-to-text application, not just single-letter recognition.

## Model Architecture

\`\`\`
Input (96x96x3)
  -> Rescaling ([0,1] -> [-1,1] for MobileNetV2)
  -> MobileNetV2 (ImageNet pretrained, called in inference mode to keep BatchNorm stable)
  -> GlobalAveragePooling2D
  -> Dense(256, relu)
  -> Dropout(0.4)
  -> Dense(29, softmax)
\`\`\`

Training happens in two phases within a single training cell:
1. **Phase 1 — head training** (base frozen): only the classifier head trains
2. **Phase 2 — fine-tuning** (top ~30 layers unfrozen): trained with a much lower learning rate (1e-5) to adapt pretrained features to ASL hand shapes specifically, while BatchNorm layers stay in inference mode to avoid destabilizing training

## How to Run

1. Open `asl_alphabet_recognition.ipynb` in Google Colab.
2. Get a Kaggle API token from [kaggle.com/settings/api](https://www.kaggle.com/settings/api) ("Create New Token").
3. Run cells in order:
   - **Cells 1–3**: install dependencies, authenticate with Kaggle (token pasted via a secure prompt), download the dataset
   - **Cells 4–6**: configure training settings, build the stratified data split, build the `tf.data` pipeline with augmentation
   - **Cells 7–8**: build the MobileNetV2 model, then train (Phase 1 + Phase 2 fine-tuning, 10 epochs total)
   - **Cell 9**: plot training curves
   - **Cell 10**: evaluate on the held-out test set (accuracy, precision, recall, F1, confusion matrix)
   - **Cell 11**: save the trained model + class labels to Google Drive
   - **Cell 12**: upload an image and get a live prediction
   - **Cell 13** *(optional)*: upload and predict multiple images in one run

If your Colab runtime disconnects after training has already completed once, use **Cell 11B** to reload the saved model from Drive instead of retraining from scratch.

## Evaluation Metrics

The model is evaluated on a held-out test split using:
- Accuracy
- Precision (macro & weighted average)
- Recall (macro average)
- F1-score (macro & weighted average)
- Full per-class classification report
- Confusion matrix (visualized as a heatmap)

## Project Structure

\`\`\`
asl_alphabet_recognition.ipynb   # Full Colab notebook (data pipeline, training, evaluation, inference)
asl_alphabet_recognition.py      # Same pipeline as a plain script (# %% cell markers)
README.md                        # This file
\`\`\`

## Notes & Limitations

- The ASL Alphabet dataset consists of images captured in a fairly consistent setting (plain background, close-up framing, similar lighting). As a result, real-world uploaded images with different backgrounds, lighting, or hand distance from the camera can be harder for the model to classify correctly than the dataset's own held-out test split would suggest. For best results, use images with the hand filling most of the frame against a plain, well-lit background.
- `SUBSET_PER_CLASS` (Cell 4) and the epoch split (`PHASE1_EPOCHS` / `PHASE2_EPOCHS` in Cell 8) can be increased for higher accuracy at the cost of longer training time.
- Image augmentation (`RandomBrightness`) is explicitly configured with `value_range=(0.0, 1.0)` to match the pipeline's [0,1]-normalized images — using the default range would corrupt training images and cause the model to fail to learn from them.

## Acknowledgements

- Dataset: [grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) on Kaggle
- Base architecture: [MobileNetV2](https://arxiv.org/abs/1801.04381) (ImageNet pretrained weights via `tf.keras.applications`)
