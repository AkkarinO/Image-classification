from pandas import DataFrame
from itertools import product
import numpy as np
from PIL import Image
from skimage.feature import greycomatrix, greycoprops
from glob import glob
from os.path import sep, join, splitext

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

feature_names = ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']

distances = (1, 3, 5)

angles = (0, np.pi/4, np.pi/2, 3*np.pi/4)

def get_full_names():
    dist_str = ('1', '2', '5')
    angles_str = '0deg, 45deg, 90deg, 135deg'.split(',')
    return ['_'.join(f) for f in product(feature_names, dist_str, angles_str)]


# GLCM matrix calculation
def get_glcm_feature_array(patch):
    patch_64 = (patch / np.max(patch) * 63).astype('uint8')
    glcm = greycomatrix(patch_64, distances, angles, 64, True, True)
    feature_vector = []
    for feature in feature_names:
        feature_vector.extend(list(greycoprops(glcm, feature).flatten()))
    return feature_vector


# Files with photos
texture_folder = "Zdjęcia_final"
samples_folder = "Wycinki"
paths = glob(texture_folder + "\\*\\*.jpg")
fil2 = [p.split(sep) for p in paths]
_, categories, files = zip(*fil2)
size = 128, 128 # image proportion

features = []
for category, infile in zip(categories, files):
    img = Image.open(join(texture_folder, category, infile))
    xr = np.random.randint(0, img.width-size[0], 10)            # choosing coordinates for 10 samples
    yr = np.random.randint(0, img.height-size[1], 10)
    base_name, _ = splitext(infile)                             # extracting names from the image
    for i, (x, y) in enumerate(zip(xr, yr)):
        img_sample = img.crop((x, y, x+size[0], y+size[1]))
        img_sample.save(join(samples_folder, category, f'{base_name:s}_{i:02d}.jpg'))   # saving patch to the file
        img_grey = img.convert('L')                             # conversion to gray scale
        feature_vector = get_glcm_feature_array(np.array(img_grey))     # generating texture features
        feature_vector.append(category)                         # adding category name to the vector
        features.append(feature_vector)                         # adding names vector to the feature vector

full_feature_names = get_full_names()
full_feature_names.append('Category')

df = DataFrame(data=features, columns=full_feature_names)
df.to_csv('textures_data.csv', sep=',', index=False)        # saving data to csv file

features = pd.read_csv('textures_data.csv', sep=',')    # opening file with images features

data = np.array(features)       # converting data to the array
x = (data[:, :-1]).astype('float64')    # saving columns without the last one
y = data[:, -1]

x_transform = PCA(n_components=3)
xt = x_transform.fit_transform(x)

red = y == 'Cegła'
blue = y == 'Tynk'
green = y == 'Tynk mozaikowy'

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xt[red, 0], xt[red, 1], xt[red, 2], c="r")
ax.scatter(xt[blue, 0], xt[blue, 1], xt[blue, 2], c="b")
ax.scatter(xt[green, 0], xt[green, 1], xt[green, 2], c="g")

classifier = svm.SVC(gamma='auto')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(acc)

cm = confusion_matrix(y_test, y_pred, normalize='true')
print(cm)

disp = plot_confusion_matrix(classifier, x_test, y_test, cmap=plt.cm.Blues)
plt.show()