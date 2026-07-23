# Energy-Efficient Pattern Recognition Using Spiking Neural Networks

## Overview

This project investigates the use of **Spiking Neural Networks (SNNs)** as an energy-efficient alternative to traditional **Convolutional Neural Networks (CNNs)** for image classification. The project compares different neural network architectures based on classification accuracy, inference latency, and energy consumption.

The implementation includes training, evaluation, visualization of results, and testing on both standard and custom datasets.

---

## Objectives

- Develop and evaluate CNN and SNN models for image classification.
- Compare the performance of different architectures.
- Measure energy consumption and inference latency.
- Analyze the trade-off between accuracy and computational efficiency.

---

## Features

- CNN and Spiking Neural Network (SNN) implementations
- Real-time handwritten digit recognition using a webcam
- Automatic Region of Interest (ROI) extraction
- Image preprocessing and digit normalization
- Prediction comparison across multiple neural network architectures
- Confidence score reporting
- Inference latency measurement
- Energy consumption analysis
- Accuracy and performance visualization
---

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pickle

---

## Project Structure

```
├── cnn_model.py
├── conv_snn_model.py
├── fc_snn_model.py
├── train_models.py
├── evaluate_models.py
├── data_loader.py
├── test_image.py
├── webcam.py
├── plot.py
├── plot_results.py
├── plot_timesteps.py
├── timestep_experiment.py
├── custom_dataset/
├── *.pth
├── accuracy_comparison.png
├── energy_vs_accuracy.png
├── timestep_accuracy.png
├── timestep_energy.png
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/Nikhil-Panchal29/Energy-Efficient-Pattern-Recognition-Using-Spiking-Neural-Networks.git
```

Move into the project directory

```bash
cd Energy-Efficient-Pattern-Recognition-Using-Spiking-Neural-Networks
```

Install dependencies

```bash
pip install torch torchvision matplotlib numpy
```

---

## Running the Project

Train the models

```bash
python train_models.py
```

Evaluate trained models

```bash
python evaluate_models.py
```

Test using an image

```bash
python test_image.py
```

Run webcam prediction

```bash
python webcam.py
```

---

## Results

The project evaluates different neural network architectures using:

- Classification Accuracy
- Inference Latency
- Energy Consumption

Generated visualizations include:

- Accuracy Comparison
- Energy vs Accuracy
- Timestep Accuracy
- Timestep Energy

---

## Dataset

The project includes a custom dataset organized into multiple class folders.

---

## Webcam Demo
<img width="1600" height="1041" alt="WhatsApp Image 2026-07-23 at 12 33 54 PM" src="https://github.com/user-attachments/assets/f366e3bd-ed82-4eff-aa61-e9ee52c7cab2" />
<img width="1600" height="1041" alt="WhatsApp Image 2026-07-23 at 12 33 54 PM (1)" src="https://github.com/user-attachments/assets/53342b61-894b-4e99-a1a9-865bf0cfee4e" />

## Performance Comparison

### Accuracy Comparison

<img width="1920" height="1440" alt="image" src="https://github.com/user-attachments/assets/29a63636-be82-43b0-9fc6-af977a8902b9" />


### Energy vs Accuracy

<img width="1920" height="1440" alt="image" src="https://github.com/user-attachments/assets/d57da653-e6e2-42ca-9f96-142677b0690c" />


### Timestep Accuracy

<img width="1920" height="1440" alt="image" src="https://github.com/user-attachments/assets/1bf1f089-556e-4ada-99bc-686cfd1c233d" />


### Timestep Energy

<img width="1920" height="1440" alt="image" src="https://github.com/user-attachments/assets/deec4516-95b2-4f5d-9319-3dbd431f0464" />


## Contributors

This project was developed as a collaborative major project.

---

## Future Improvements

- Train on larger datasets
- Improve SNN accuracy
- Optimize energy consumption further
- Deploy as a real-time application
- Support additional neuromorphic hardware

---

## License

This project is intended for academic and educational purposes.
