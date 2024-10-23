
import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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


def load_flat_imgs(folder):

    flattened_images = [] # Initialize an empty list to store flattened images

    # Loop over all files in the directory
    for filename in os.listdir(folder):
        image = Image.open(os.path.join(folder, filename)).convert('L') # Open the image, convert to grayscale
        flattened_image = np.array(image).flatten() # Flatten the image and convert it to a numpy array
        flattened_images.append(flattened_image) # Append the flattened image to the list

    return np.vstack(flattened_images) # Stack all flattened images into a 2D matrix (samples x features)


def load_imgs(folder):

    images = [] # Initialize an empty list to store images

    # Loop over all files in the directory
    for filename in os.listdir(folder):
        image = Image.open(os.path.join(folder, filename)).convert('L') # Open the image, convert to grayscale
        image = np.array(image) # Convert the image to a numpy array
        images.append(image) # Append the image to the list

    return np.stack(images) # Stack all images into a 3D matrix (samples x height x width)