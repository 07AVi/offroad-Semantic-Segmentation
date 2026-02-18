# Offroad Terrain Semantic Segmentation

## Overview
This project implements semantic segmentation using DeepLabV3+ with a ResNet34 encoder.

The model performs pixel-level classification on offroad terrain images and segments them into 6 different classes.



## Model Details
- Architecture: DeepLabV3+
- Encoder: ResNet34 (ImageNet pretrained)
- Framework: PyTorch
- Number of Classes: 6
- Optimizer: Adam
- Learning Rate: 0.0001
- Loss Function: CrossEntropyLoss



## Training Performance
The model was trained for 13 epochs.

Final Validation IoU: ~0.85  
Loss decreased steadily from 0.63 to 0.35.



## How to Run

Install dependencies:

pip install -r requirements.txt

Train the model:

python train.py

Generate predictions:

python predict.py

Evaluate model:

python evaluate.py



## Conclusion
The model achieves strong segmentation performance and demonstrates effective terrain understanding for offroad images.
