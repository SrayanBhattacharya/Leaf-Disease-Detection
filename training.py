import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Training image processing

trainingSet = tf.keras.utils.image_dataset_from_directory(
    "train",
    labels = "inferred",
    label_mode = "categorical",
    class_names = None,
    color_mode = "rgb",
    batch_size = 32,
    image_size = (256, 256),
    shuffle = True,
    seed = None,
    validation_split = None,
    subset = None,
    interpolation = "bilinear",
    follow_links = False,
    crop_to_aspect_ratio = False,
    pad_to_aspect_ratio = False,
)

#Validation image processsing

validationSet = tf.keras.utils.image_dataset_from_directory(
    "valid",
    labels = "inferred",
    label_mode = "categorical",
    class_names = None,
    color_mode = "rgb",
    batch_size = 32,
    image_size = (256, 256),
    shuffle = True,
    seed = None,
    validation_split = None,
    subset = None,
    interpolation = "bilinear",
    follow_links = False,
    crop_to_aspect_ratio = False,
    pad_to_aspect_ratio = False,
)