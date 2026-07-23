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

- CNN implementation using PyTorch
- Convolutional Spiking Neural Network (SNN)
- Fully Connected Spiking Neural Network
- Training and evaluation scripts
- Testing on custom image datasets
- Webcam-based image prediction
- Accuracy, latency, and energy comparison
- Performance visualization using graphs

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
