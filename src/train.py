
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from tensorflow import keras
from keras.applications import MobileNetV2 as MNV2
from keras import layers, models
from keras.callbacks import ModelCheckpoint, EarlyStopping
from joblib import Parallel, delayed
from joblib import parallel_backend
import datetime

from src.data_loader import *


def define_model(model_name="VGG16", input_shape=(130, 180, 1), num_classes=5):
    """
    Define a model using a pretrained CNN architecture adapted for grayscale input.
    
    Parameters:
        model_name (str): The name of the pretrained model to use. Options: 'VGG16', 'ResNet50', 
                          'MobileNetV2', 'EfficientNetB0', 'LeNet'.
        input_shape (tuple): The shape of the input images, e.g., (224, 224, 1) for grayscale.
        num_classes (int): The number of output classes.
    
    Returns:
        model (tf.keras.Model): The constructed and compiled model.
    """
    base_model = None

    if model_name == "VGG16":
        base_model = tf.keras.applications.VGG16(
            weights="imagenet", include_top=False, input_shape=(input_shape[0], input_shape[1], 3)
        )
    elif model_name == "ResNet50":
        base_model = tf.keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_shape=(input_shape[0], input_shape[1], 3)
        )
    elif model_name == "MobileNetV2":
        base_model = tf.keras.applications.MobileNetV2(
            weights="imagenet", include_top=False, input_shape=(input_shape[0], input_shape[1], 3)
        )
    elif model_name == "EfficientNetB0":
        base_model = tf.keras.applications.EfficientNetB0(
            weights="imagenet", include_top=False, input_shape=(input_shape[0], input_shape[1], 3)
        )
    elif model_name == "LeNet":
        # LeNet is not a pretrained model but a simple CNN architecture.
        model = models.Sequential([
            layers.Conv2D(32, (5, 5), activation="relu", input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (5, 5), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(120, activation="relu"),
            layers.Dense(84, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Add a preprocessing layer for grayscale to RGB conversion
    inputs = layers.Input(shape=input_shape)
    if input_shape[2] == 1:
        x = layers.Conv2D(3, (3, 3), padding="same", name="grayscale_to_rgb")(inputs)
    else:
        x = inputs

    # Use the base model
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    # Construct and compile the model
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def train_CV(
    model_name, 
    model_desc=None, 
    k=5, 
    EPOCHS=10, 
    BATCH_SIZE=32, 
    results_path='../results/cv_scores.csv', 
    patience=3, 
    save_model_dir='../results/models',
    n_jobs=-1  # Use all available cores by default
):

    # Load the data
    X_train, y_train = load_data('../data/processed/train/images')  # Customize load_data as needed

    # Initialize KFold cross-validator
    kf = KFold(n_splits=k, shuffle=True, random_state=22)
    os.makedirs(save_model_dir, exist_ok=True)

    def train_single_fold(fold, train_index, val_index):
        print(f"Training fold {fold + 1}/{k}...")

        # Split the data into training and validation sets for the fold
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Create a fresh model for this fold
        model_fold = define_model(model_name)

        # Define model save path for the fold
        model_save_path = os.path.join(save_model_dir, f"{model_desc}_fold{fold + 1}_best_model.keras")

        # Define callbacks
        checkpoint = ModelCheckpoint(
            filepath=model_save_path, 
            monitor='val_loss', 
            save_best_only=True, 
            verbose=1
        )
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=patience, 
            restore_best_weights=True
        )

        # Train the model
        model_fold.fit(
            X_train_fold,
            y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0,  # Suppress verbose output for parallel folds
            callbacks=[checkpoint, early_stopping]
        )

        # Predict and calculate metrics
        y_pred_fold = model_fold.predict(X_val_fold)
        y_pred_labels = np.argmax(y_pred_fold, axis=1)
        y_true_labels = np.argmax(y_val_fold, axis=1)

        acc = accuracy_score(y_true_labels, y_pred_labels)
        f1 = f1_score(y_true_labels, y_pred_labels, average="weighted")

        print(f"Fold {fold + 1} completed with Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
        return acc, f1, model_save_path

    results = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        results.append(train_single_fold(fold, train_idx, val_idx))


    # Aggregate results
    fold_accuracies, fold_f1scores, saved_model_paths = zip(*results)

    # Default model description if none provided
    if model_desc is None:
        model_desc = model_name

    # Save results to CSV
    if not os.path.exists(results_path):
        with open(results_path, 'w') as f:
            f.write('model,accuracy,f1\n')

    with open(results_path, 'a') as f:
        f.write(f'{model_desc},{np.mean(fold_accuracies)},{np.mean(fold_f1scores)}\n')

    return {
        "model": model_desc,
        "accuracy": np.mean(fold_accuracies),
        "f1_score": np.mean(fold_f1scores),
        "saved_model_paths": saved_model_paths
    }