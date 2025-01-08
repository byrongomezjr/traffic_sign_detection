# Traffic Sign Detection  🚷 ⛔️

A deep learning project using Keras to detect traffic signs, specifically focusing on stop signs and traffic lights.

## Project Structure

- `src/`: Source code for the project.
  - `train.py`: Script for training the model.
  - `predict.py`: Script for making predictions with the trained model.
  - `utils.py`: Utility functions for data preprocessing and visualization.
- `model/`: Directory for storing the trained model.
- `data/`: Directory for storing the dataset.
- `README.md`: This file.

## Getting Started

1. Clone the repository.
2. Install the required dependencies.
3. Run `python src/train.py` to train the model.
4. Run `python src/predict.py` to make predictions with the trained model.

## Notes

- The dataset used is the German Traffic Sign Recognition Benchmark (GTSRB).
- The model is trained on the stop signs and traffic lights classes.
- The model is saved in the `model/` directory.
- The test images are in the `data/test/` directory.
- The training images are in the `data/train/` directory.
