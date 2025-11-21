# Fake News Detection using LSTM

## Project Overview

This project implements a **Fake News Detection System** using a **Long Short-Term Memory (LSTM) Neural Network**.
The goal is to classify news headlines as either **'Fake'** or **'Not Fake'** based on their textual content.
Text preprocessing techniques, such as stemming, removing stop words, and one-hot encoding, are applied before feeding the data into the LSTM model.

## Dataset

The dataset used for this project is sourced from Kaggle: [Fake News Dataset](https://www.kaggle.com/datasets/ronikdedhia/fake-news).

It contains news articles with the following columns:

*   `id`: Unique identifier for each news article.
*   `title`: The title of the news article.
*   `author`: The author of the news article.
*   `text`: The full text of the news article.
*   `label`: Binary label indicating whether the news is fake (1) or real (0).

## Technologies and Libraries

This project leverages several Python libraries for data manipulation, text preprocessing, and deep learning:

*   **pandas**: For data manipulation and analysis.
*   **numpy**: For numerical operations.
*   **matplotlib**: For plotting and data visualization.
*   **seaborn**: For enhanced statistical data visualization.
*   **wordcloud**: To generate word cloud visualizations of frequently occurring words.
*   **tensorflow**: The primary deep learning framework, especially `tf.keras` for building and training the LSTM model.
*   **nltk**: Natural Language Toolkit for text preprocessing tasks like stop word removal and stemming.
*   **re**: Python's regular expression module for text cleaning.
*   **scikit-learn (sklearn)**: For model selection (train-test split) and evaluation metrics (classification report, confusion matrix, accuracy score, ROC curve, AUC).

## Data Preprocessing Steps

1.  **Data Loading**: The dataset is loaded into a pandas DataFrame.
2.  **Handling Missing Values**: Rows with any missing values in the `title`, `author`, or `text` columns are dropped to ensure data quality.
3.  **Feature Separation**: The `label` column is separated as the dependent variable (Y), and the remaining columns (excluding `label`) form the independent features (X).
4.  **Text Cleaning**: The `title` column undergoes a series of cleaning steps:
    *   Removal of non-alphabetic characters.
    *   Conversion to lowercase.
    *   Tokenization (splitting into words).
    *   Removal of English stop words.
    *   Stemming using `PorterStemmer` to reduce words to their root form.
5.  **One-Hot Representation**: Each cleaned news title is converted into a one-hot encoded representation, mapping words to numerical indices based on a predefined vocabulary size (5000 in this project).
6.  **Padding**: The one-hot encoded sequences are padded to a fixed length (20 words) to ensure uniform input for the neural network's embedding layer.

## Model Architecture

The deep learning model is built using `tf.keras.Sequential` and consists of the following layers:

1.  **Embedding Layer**: Maps each word index to a dense vector of fixed size (`embedding_vector_features = 40`).
2.  **Dropout Layer (0.3)**: Regularization layer to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.
3.  **LSTM Layer**: A Long Short-Term Memory layer with 100 units, capable of learning long-term dependencies in sequential data.
4.  **Dropout Layer (0.3)**: Another dropout layer for further regularization.
5.  **Dense Output Layer**: A fully connected layer with a single neuron and a `sigmoid` activation function for binary classification.

The model is compiled with `binary_crossentropy` as the loss function and `adam` as the optimizer, aiming to maximize `accuracy`.

## Training and Evaluation

### Training

The preprocessed data is split into training and testing sets (67% training, 33% testing). The model is trained for 10 epochs with a batch size of 64. Training and validation loss and accuracy are tracked during this process.

### Performance Metrics

The model's performance is evaluated using standard classification metrics on the test set:

*   **Classification Report**: Provides precision, recall, and F1-score for each class (fake/not fake).
*   **Confusion Matrix**: Shows the counts of true positives, true negatives, false positives, and false negatives.
*   **Accuracy Score**: The overall accuracy of the model.
*   **ROC Curve and AUC Score**: Visualizes the trade-off between the True Positive Rate (TPR) and False Positive Rate (FPR) across various threshold settings, with the Area Under the Curve (AUC) providing a single metric of the model's discriminative power.

### Results

Initial training results (without dropout):

*   **Epoch 10**: `accuracy: 0.9963`, `loss: 0.0137`, `val_accuracy: 0.9082`, `val_loss: 0.5130`

Results after adding Dropout layers:

*   **Classification Report**:
    ```
                  precision    recall  f1-score   support

           0       0.57      0.61      0.59      3419
           1       0.44      0.40      0.41      2616

    accuracy                           0.52      6035
   macro avg       0.50      0.50      0.50      6035
weighted avg       0.51      0.52      0.51      6035
    ```
*   **Confusion Matrix**:
    ```
    [[2081 1338]
     [1581 1035]]
    ```
*   **Accuracy Score**: `0.5163`
*   **AUC Score**: `0.47` (This indicates the model is performing worse than random guessing. Further hyperparameter tuning and model architecture changes are needed.)

**Note**: The initial training without dropout showed very high training accuracy and low training loss but a significant difference from validation metrics, indicating overfitting. After adding dropout, the validation performance decreased. This suggests the model might be too complex for the given data or further optimization of hyperparameters (learning rate, LSTM units, dropout rates, embedding size, `sent_length`) is required, or potentially using a more robust dataset or preprocessing strategy like word embeddings (Word2Vec, GloVe) instead of simple one-hot encoding for better semantic understanding. It is also important to consider that only the `title` column was used for text processing; including the `text` column could significantly improve performance.
```
