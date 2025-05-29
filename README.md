# Isolated Sign Language Recognition with Pose Estimation and Transformers

This repository implements an end-to-end pipeline for isolated sign language recognition using MediaPipe for pose/face/hand landmark extraction and a Transformer-based model for classification. It supports data preprocessing, augmentation, normalization, training, evaluation and real-time prediction.

## Table of Contents
1. [Features](#features)  
2. [Project Structure](#project-structure)  
3. [Installation](#installation)  
4. [Usage](#usage)  
   - [Data Preprocessing](#data-preprocessing)  
   - [Training](#training)  
   - [Evaluation](#evaluation)  
   - [Real-time Prediction](#real-time-prediction)  
5. [Implemented Tools](#implemented-tools)  
6. [Dependencies](#dependencies)  
7. [License](#license)  

## Features
- Extracts hand, pose and face landmarks using MediaPipe.  
- Applies data augmentation (rotation, zoom, shift, re-anchoring wrists).  
- Normalizes sequences to a consistent body reference frame.  
- Trains a Transformer classifier on landmark sequences.  
- Supports early stopping and metric tracking.  
- Provides evaluation scripts with precision/recall/F1 and confusion matrix display.  
- Real-time webcam prediction with overlaid text.  
- Utility to generate side-by-side landmark comparison videos.

## Project Structure
```
README.md  
EDA.ipynb  
Process_landmarks.ipynb  
src/  
 ├── evaluate.py  
 ├── real_time_prediction.py  
 ├── train.py  
 ├── model/  
 │    ├── dataloader.py  
 │    ├── early_stopping.py  
 │    ├── evaluator.py  
 │    ├── trainner.py  
 │    └── transformer.py  
 ├── preprocessing/  
 │    ├── __init__.py  
 │    ├── data_augmentation.py  
 │    ├── feature_extraction.py  
 │    ├── interpolation.py  
 │    ├── normalization.py  
 │    └── video_processor.py  
 └── utils/  
      └── utils.py  
```

## Implemented Tools
- MediaPipe landmark detection (`feature_extraction.py`)  
- Data augmentation: rotation, zoom, shift, re-anchor (`data_augmentation.py`)  
- Coordinate normalization to body frame (`normalization.py`)  
- Temporal interpolation (`interpolation.py`)  
- PyTorch Transformer model & training (`transformer.py`, `trainner.py`)  
- Early stopping callback (`early_stopping.py`)  
- Evaluation metrics & plots (`evaluator.py`)  
- Webcam demo with overlay (`real_time_prediction.py`)  
- Video comparison utility (`utils/utils.py`)

## Installation
```bash
git clone https://github.com/your-org/Sign-Language-Translator-1.git
cd Sign-Language-Translator-1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
### Data Preprocessing
This part of the pipeline can be performed using the provided preprocessing notebook (`Process_landmarks.ipynb`). The notebook guides you through extracting landmarks with `LandmarkExtractor` (`src/preprocessing/feature_extraction.py`), applying temporal interpolation (`SequenceInterpolator`), data augmentation (`DataAugmentor`), and normalization (`LandmarkNormalizer`) to prepare your dataset for training.

### Training
```bash
python src/train.py \
  --data-dir path/to/landmark_sequences \
  --output-dir results/my_experiment \
  --epochs 50 \
  --batch-size 32
```

### Evaluation
```bash
python src/evaluate.py \
  --model-path results/my_experiment/model.pth \
  --test-data path/to/test_sequences
```

### Real-time Prediction
```bash
python src/real_time_prediction.py \
  --vocab vocab_words.json \
  --model-path results/my_experiment/model.pth \
  --device cuda
```

## Dependencies
- Python 3.7+  
- mediapipe  
- opencv-python  
- numpy  
- torch  
- scikit-learn  
- matplotlib  
- zarr  
- faiss  

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
