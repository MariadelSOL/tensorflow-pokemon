from copy import deepcopy
from keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters.rank import median
from skimage.morphology import disk
import skimage
import skimage.io
import skimage.filters
import skimage.transform
import skimage.feature
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy.linalg as sla
# %matplotlib inline
from keras.models import load_model
from keras.layers import Dense, Activation, Flatten, Input, Dropout, GlobalAveragePooling2D
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.applications import ResNet50 # resnet50
input_tensor = Input(shape=(224, 224, 3))
resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=input_tensor)
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

# %%time
faces = list()
paths = list()
origin_faces = list()
for path in os.listdir('faces/out'):
    paths.append(path)
    img = skimage.io.imread('faces/out/' + path)
    origin_faces.append(img)
    img = skimage.color.rgb2gray(img)
    img = skimage.color.gray2rgb(img)
    faces.append(img)
data = np.array(faces).astype(np.float64)
data = preprocess_input(data)
faces_len = len(data)
prediction_faces = resnet.predict(data).reshape(faces_len, 2048)

# %%time
pokemons = list()
origin_pokemons = list()
pokepaths = list()
dir_ = 'pokefaces'
for path in os.listdir(dir_):
    path = os.path.join(dir_, path)
    if '.png' not in path:
        continue
    pokepaths.append(path)
    pokeimg = skimage.io.imread(path)
    origin_pokemons.append(skimage.io.imread(path))
    pokeimg = skimage.transform.resize(pokeimg, (224, 224, 3))
    pokeimg = skimage.color.rgb2gray(pokeimg)
    pokeimg = skimage.color.gray2rgb(pokeimg)
    pokemons.append(pokeimg)
data = np.array(pokemons).astype(np.float64)
data = preprocess_input(data)
poke_len = len(data)
prediction_poke = resnet.predict(data).reshape(poke_len, 2048)

distances = np.zeros((faces_len, poke_len))
for face_idx, face in enumerate(prediction_faces):
    for poke_idx, poke in enumerate(prediction_poke):
        distances[face_idx, poke_idx] = np.dot(face, poke) / sla.norm(poke) / sla.norm(face)

import seaborn as sns
sns.heatmap(distances)

from collections import Counter

# np.argsort(distances[10])[::-1]

#plt.subplots(figsize=(12, face_len * 4))
sns.set_style("whitegrid", {'axes.grid' : False})
poke_cnt = Counter()
image_pairs = []
for idx in range(faces_len):
    priority = np.argsort(distances[idx])[::-1]
    for i, p in enumerate(priority):
        if poke_cnt[p] >=5:
            continue
        else:
            poke_cnt[p] += 1
            poke_idx = p
            break
    image_pairs.append((idx, poke_idx))

np.random.shuffle(image_pairs)
for idx, poke_idx in image_pairs:
    plt.figure(figsize=(5, 2))
    plt.subplot(121)
    plt.imshow(origin_faces[idx])
    plt.subplot(122)
    plt.imshow(origin_pokemons[poke_idx])
    # plt.show()
    plt.savefig('data.png')
