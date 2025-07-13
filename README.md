# CNN for MNIST Digit Classification

This project uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify handwritten digits from the MNIST dataset (digits 0 to 9).

---

## Dataset

- Source: Kaggle MNIST digit recognizer competition
- Image size: 28x28 grayscale
- Total classes: 10 (digits 0 to 9)

---

## Model Architecture

The CNN model follows this architecture:

Input → [Conv2D → ReLU] x2 → MaxPool → Dropout → [Conv2D → ReLU] x2 → MaxPool → Dropout → Flatten → Dense → Dropout → Output (Softmax)


---

## Data Preparation

- Normalized pixel values to range [0, 1]
- Reshaped to shape `(28, 28, 1)` for CNN input
- Labels converted to one-hot encoded vectors

---

## Data Splits

- The dataset was initially split into train and test
- The training set was further split into:
  - Training set (90%)
  - Validation set (10%)
- This helps monitor performance and avoid overfitting

---

## Training Setup

- Optimizer: RMSprop
- Loss function: Categorical Crossentropy
- Learning rate reduced using ReduceLROnPlateau
- Data augmented using Keras `ImageDataGenerator`

---

## Results

- Final training accuracy: 100.00%
- Final validation accuracy: 99.36%
- Confusion matrix used for error analysis
- model is not overfitting (i.e. validation accuracy is close to training accuracy) ~ Dataset is small and well labled

---

## Submission

Predictions were made on the test set and saved as a CSV in the required Kaggle format with:

- `ImageId` column (1 to 28000)
- `Label` column (predicted digit)

---

## How to Use

1. Load and preprocess the dataset
2. Define and train the CNN model
3. Evaluate using validation data
4. Predict on test data
5. Save predictions in submission format

---

## References

- My Kaggle Notebook: https://www.kaggle.com/code/furyfist/digit-recognizer
- Referenced Notebook: https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

---

## Notes

- The validation set helps detect overfitting early by showing how the model performs on unseen data.
- Data augmentation improves generalization by exposing the model to various image transformations.
- This simple CNN achieves strong performance without using complex architectures.

