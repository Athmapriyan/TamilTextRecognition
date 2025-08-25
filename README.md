# Tamil Handwritten Character Recognition

## Overview

This project implements a deep learning model to recognize handwritten Tamil characters. Using a Convolutional Neural Network (CNN), the model is trained on a custom dataset of handwritten characters and can accurately classify new, unseen images.

## Features

-   **Data Loading and Preprocessing**: The project demonstrates how to load image data from a directory, preprocess it by resizing and normalizing, and prepare it for model training.
-   **CNN Model Architecture**: A custom CNN model is designed and built using TensorFlow/Keras for robust image classification.
-   **Model Training and Evaluation**: The model is trained on a dedicated training set and evaluated on a separate test set to measure its performance.
-   **Interactive Character Recognition**: An interactive functionality is provided using `ipywidgets` that allows users to upload a handwritten Tamil character image and get a prediction from the trained model.

## Dataset

The dataset used for this project is the **Tamil Handwritten Character Recognition** dataset available on Kaggle. You can download it from [this link](https://www.kaggle.com/datasets/gauravduttakiit/tamil-handwritten-character-recognition).

The dataset contains:
-   **Training Set**: 50,296 grayscale images of size 64x64.
-   **Test Set**: 12,574 grayscale images of size 64x64.
-   **Classes**: 156 distinct Tamil characters.

## Setup and Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/YourUsername/tamil-character-recognition.git](https://github.com/YourUsername/tamil-character-recognition.git)
    cd tamil-character-recognition
    ```

2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the dataset**:
    -   Download the dataset from the Kaggle link provided above.
    -   Extract the contents and place the `train` and `test` folders inside a `data/` directory in the project's root folder. Your directory structure should look like this:
        ```
        tamil-character-recognition/
        ├── data/
        │   ├── train/
        │   └── test/
        ├── ...
        ```

## Usage

1.  **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

2.  Open the `tamil_character_recognition.ipynb` notebook.
3.  Run the cells sequentially to perform the following steps:
    -   Load and explore the data.
    -   Preprocess the data.
    -   Build the CNN model.
    -   Train the model.
    -   Evaluate the model's performance on the test set.
    -   Use the interactive widget to upload your own handwritten Tamil character image for recognition.

## Model Performance

The trained model achieved the following performance metrics:
-   **Training Accuracy**: >99%
-   **Validation Accuracy**: ~93%
-   **Test Accuracy**: ~93%

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

**Feel free to connect with me:**
-   [Your LinkedIn Profile]
-   [Your Personal Website/Portfolio]
