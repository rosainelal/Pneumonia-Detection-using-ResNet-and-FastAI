# Pneumonia Detection Model

## Overview
This project implements a pneumonia detection system that classifies chest X-ray images into three categories:
- Normal
- Viral Pneumonia
- Bacterial Pneumonia

The model is trained using **ResNet** and **FastAI** to achieve high accuracy in detecting pneumonia types. The trained model is then deployed using **Gradio** for user-friendly image-based diagnosis.

## Workflow
1. **Training the Model**
   - Run `train_model.ipynb` to train the model on a dataset containing normal, viral, and bacterial pneumonia images.
   - The model is trained using ResNet architecture and optimized with FastAI.

2. **Deploying the Model**
   - Use `final.ipynb` to load the trained model and set up a **Gradio** interface.
   - Users can upload chest X-ray images, and the model will classify them accordingly.

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install fastai gradio torch torchvision
```

## Running the Project
1. **Train the Model:**
   - Open `train_model.ipynb` in Jupyter Notebook or Google Colab.
   - Train the model and save the best weights.

2. **Run the Gradio Interface:**
   - Open `final.ipynb` and execute the cells to launch the interface.
   - Upload an X-ray image to get the classification result.

## Dataset
The dataset consists of labeled chest X-ray images categorized as **normal, viral pneumonia,** and **bacterial pneumonia.** Ensure the dataset is structured properly before training.

## Model Architecture
- **ResNet:** A deep convolutional neural network used for feature extraction.
- **FastAI:** Simplifies training and optimization with built-in functions for image classification.

## Future Improvements
- Enhancing accuracy with data augmentation techniques.
- Implementing attention mechanisms for better feature extraction.
- Deploying the model as a web application.

## Acknowledgments
Special thanks to open-source datasets and libraries that made this project possible.
