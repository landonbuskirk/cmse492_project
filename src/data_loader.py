
import numpy as np
import pandas as pd
from PIL import Image
import os


def load_img_data(train=True):

    img_data = np.array([])

    if train:
        for ind in pd.read_csv('../data/processed/train/labels.csv')['image_id']:
            image = Image.open(f'../data/processed/train/images/{ind}.jpg')
            img_data = np.append(np.array(image) / 255.0, img_data)

    else:
        for ind in pd.read_csv('../data/processed/test/labels.csv')['image_id']:
            image = Image.open(f'../data/processed/test/images/{ind}.jpg')
            img_data = np.append(np.array(image) / 255.0, img_data)

    return img_data