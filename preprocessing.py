import numpy as np
import skimage as sk
import os
import matplotlib.pyplot as plt

PATH = 'C:/Users/ksb08/PycharmProjects/DogFaceNet/data/dogfacenet/aligned/after_4_bis'
SIZE = (224,224,3)

filenames = np.empty(0)
labels = np.empty(0)
idx = 0
for root,dirs,files in os.walk(PATH):
    if len(files)>1:
        for i in range(len(files)):
            files[i] = root + '/' + files[i]
        filenames = np.append(filenames,files)
        labels = np.append(labels,np.ones(len(files))*idx)
        idx += 1
# print(len(labels))

h,w,c = SIZE
images = np.empty((len(filenames),h,w,c))
for i,f in enumerate(filenames):
    images[i] = sk.io.imread(f)

for i,f in enumerate(filenames):
    idxБ = f.find('Б')
    if idxБ>=0:
        f = f[:idxБ] + "ü" + f[idxБ+1:]
    idxmarley = f.find('marley-groсman')
    if idxmarley>=0:
        f = f[:idxmarley] + "marley-großman" + f[idxmarley+len("marley-großman"):]
    images[i] = sk.io.imread(f)

# Normalization
images /= 255.0

nbof_classes = len(np.unique(labels))
# print(nbof_classes)

# Data Split
nbof_test = int(0.1 * nbof_classes)

keep_test = np.less(labels,nbof_test)
# print(keep_test)

images_test = images[keep_test]
labels_test = labels[keep_test]
# print(images_test)
# print(labels_test)

NBOF_PAIRS = len(images_test)
# Create pairs
h, w, c = SIZE
pairs = np.empty((NBOF_PAIRS * 2, h, w, c))
issame = np.empty(NBOF_PAIRS)
class_test = np.unique(labels_test)
for i in range(NBOF_PAIRS):
    alea = np.random.rand()
    # Pair of different dogs
    if alea < 0.5:
        # Chose the classes:
        class1 = np.random.randint(len(class_test))
        class2 = np.random.randint(len(class_test))
        while class1 == class2:
            class2 = np.random.randint(len(class_test))

        # Extract images of this class:
        images_class1 = images_test[np.equal(labels_test, class1)]
        images_class2 = images_test[np.equal(labels_test, class2)]

        # Chose an image amoung these selected images
        pairs[i * 2] = images_class1[np.random.randint(len(images_class1))]
        pairs[i * 2 + 1] = images_class2[np.random.randint(len(images_class2))]
        issame[i] = 0
    # Pair of same dogs
    else:
        # Chose a class
        clas = np.random.randint(len(class_test))
        images_class = images_test[np.equal(labels_test, clas)]

        # Select two images from this class
        idx_image1 = np.random.randint(len(images_class))
        idx_image2 = np.random.randint(len(images_class))
        while idx_image1 == idx_image2:
            idx_image2 = np.random.randint(len(images_class))

        pairs[i * 2] = images_class[idx_image1]
        pairs[i * 2 + 1] = images_class[idx_image2]
        issame[i] = 1

# print(issame)
# print(pairs)

# Test: Check the pairs
s = 10
n = 5
print(issame[s:(n+s)])
fig = plt.figure(figsize=(5,3*n))
for i in range(s,s+n):
    plt.subplot(n,2,2*(i-s)+1)
    plt.imshow(pairs[2*i]*0.5+0.5)
    plt.subplot(n,2,2*(i-s)+2)
    plt.imshow(pairs[2*i+1]*0.5+0.5)
plt.show()