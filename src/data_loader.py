
import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras


def process_raw_img_data():

    labels = pd.read_csv('../data/raw/labels.csv', index_col=0)
    labels = labels[labels['label'] != 1]
    train, test = train_test_split(labels, stratify=labels['label'], random_state=42, test_size=.2)
    train, test = train.set_index('filename'), test.set_index('filename')

    # store train and test labels into processed data folder
    train.to_csv('../data/processed/train/labels.csv')
    test.to_csv('../data/processed/test/labels.csv')

    # store train and test images into processed data folder, but lower pixel shape to (264, 200)
    for ind in train.index:
        image = Image.open(f'../data/raw/images/{ind}')
        image = image.resize((200, 150)).crop((10, 10, 190, 140))
        image.save(f'../data/processed/train/images/{ind}')

    for ind in test.index:
        image = Image.open(f'../data/raw/images/{ind}')
        image = image.resize((200, 150)).crop((10, 10, 190, 140))
        image.save(f'../data/processed/test/images/{ind}')


def load_data(folder, flat=False, test=False):

    images = [] # Initialize an empty list to store images
    labels = [] # Initialize an empty list to store labels
    if test:
        y_df = pd.read_csv('../data/processed/test/labels.csv').set_index('filename')
    else:
        y_df = pd.read_csv('../data/processed/train/labels.csv').set_index('filename')

    # Loop over all files in the directory
    for filename in os.listdir(folder):
        image = Image.open(os.path.join(folder, filename)).convert('L') # Open the image, convert to grayscale
        image = np.array(image) # Convert the image to a numpy array
        if flat:
            image = image.flatten()
        images.append(image) # Append the image to the list
        labels.append(y_df.loc[filename]['label'])

    images = np.stack(images).reshape(-1, 130, 180, 1) # Stack all images into a 4D matrix (samples x height x width x 1)
    labels = keras.utils.to_categorical(np.array(labels)-2, 5) # (samples x num_classes)

    return images, labels
