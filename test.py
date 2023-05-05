import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import skimage as sk
import os
import matplotlib.pyplot as plt
import tqdm
from sklearn.cluster import KMeans

alpha = 0.3
def triplet(y_true, y_pred):
    a = y_pred[0::3]
    p = y_pred[1::3]
    n = y_pred[2::3]
    ap = K.sum(K.square(a - p), -1)
    an = K.sum(K.square(a - n), -1)

    return K.sum(tf.nn.relu(ap - an + alpha))

def triplet_acc(y_true, y_pred):
    a = y_pred[0::3]
    p = y_pred[1::3]
    n = y_pred[2::3]
    ap = K.sum(K.square(a - p), -1)
    an = K.sum(K.square(a - n), -1)

    return K.less(ap + alpha, an)

model = tf.keras.models.load_model('C:/Users/ksb08/PycharmProjects/DogFaceNet/output/model/2023.04.29.test/dogfacenet.test.3.h5', custom_objects={'triplet':triplet,'triplet_acc':triplet_acc})

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
print('labels : ', labels, '\n')

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
# print('images : ', images, '\n')

# Normalization
images /= 255.0

nbof_classes = len(np.unique(labels))
# print(nbof_classes)

# Data Split
nbof_test = int(0.1 * nbof_classes)

keep_test = np.less(labels,nbof_test)
print('keep_test', keep_test, '\n')

images_test = images[keep_test]
labels_test = labels[keep_test]
print('images_test : ', images_test, '\n')
print('labels_test : ', labels_test, '\n')

NBOF_PAIRS = len(images_test)
# Create pairs
h, w, c = SIZE
pairs = np.empty((NBOF_PAIRS * 2, h, w, c))
issame = np.empty(NBOF_PAIRS)
class_test = np.unique(labels_test)
print('class_test : ', class_test)
for i in range(NBOF_PAIRS):
    alea = np.random.rand()
    # Pair of different dogs
    if alea < 0.5:
        # print('------------------------------ Pair of Different dog ------------------------------')
        # Chose the classes:
        class1 = np.random.randint(len(class_test))
        class2 = np.random.randint(len(class_test))
        while class1 == class2:
            class2 = np.random.randint(len(class_test))

        # Extract images of this class:
        images_class1 = images_test[np.equal(labels_test, class1)]
        images_class2 = images_test[np.equal(labels_test, class2)]
        # print('images_class1 : ', images_class1, '\n')
        # print('images_class2 : ', images_class2, '\n')

        # Chose an image amoung these selected images
        pairs[i * 2] = images_class1[np.random.randint(len(images_class1))]
        # print('pairs[i * 2] : ', pairs[i*2], np.shape(pairs[i*2]), '\n')
        pairs[i * 2 + 1] = images_class2[np.random.randint(len(images_class2))]
        # print('pairs[i * 2 + 1] : ', pairs[i*2 + 1], np.shape(pairs[i*2 + 1]), '\n')
        issame[i] = 0
        break
    # Pair of same dogs
    else:
        # print('------------------------------ Pair of Same dog ------------------------------')
        # Chose a class
        clas = np.random.randint(len(class_test))
        images_class = images_test[np.equal(labels_test, clas)]
        # print('images_class : ', images_class, '\n')

        # Select two images from this class
        idx_image1 = np.random.randint(len(images_class))
        idx_image2 = np.random.randint(len(images_class))
        while idx_image1 == idx_image2:
            idx_image2 = np.random.randint(len(images_class))

        pairs[i * 2] = images_class[idx_image1]
        # print('pairs[i * 2] : ', pairs[i*2], np.shape(pairs[i*2]), '\n')
        pairs[i * 2 + 1] = images_class[idx_image2]
        # print('pairs[i * 2 + 1] : ', pairs[i*2 + 1], np.shape(pairs[i*2 + 1]), '\n')
        issame[i] = 1
print('-------------------------------------- Pairs --------------------------------------')
print(pairs)
print(np.shape(pairs))
print('-------------------------------------- Issame --------------------------------------')
print(issame)
print(np.shape(issame))

# print(issame)
# print("-------------------------- Pairs --------------------------")
# print('pairs : ', np.shape(pairs), '\n')
# print('pairs[0] : ', np.shape(pairs[0]), '\n')
# print(pairs[0][0], '\n')
# print(pairs[0][0][0], '\n')
# print(pairs[0][0][0][0], '\n')
# plt.imshow(pairs[0])
# plt.show()

# # Test: Check the pairs
# s = 10
# n = 5
# print(issame[s:(n+s)])
# fig = plt.figure(figsize=(5,3*n))
# for i in range(s,s+n):
#     plt.subplot(n,2,2*(i-s)+1)
#     plt.imshow(pairs[2*i]*0.5+0.5)
#     plt.subplot(n,2,2*(i-s)+2)
#     plt.imshow(pairs[2*i+1]*0.5+0.5)
# plt.show()

predict = model.predict(pairs)
print("-------------------------- Predict --------------------------")
print(predict)
print(np.shape(predict))
# Separates the pairs
# emb1 = predict[0::2]
# emb2 = predict[1::2]
# print("-------------------------- emb1 --------------------------")
# print(emb1)
# print(np.shape(emb1))
# print(emb1[0])
# print("-------------------------- emb2 --------------------------")
# print(emb2)
# print(np.shape(emb2))
# print(emb2[0])
#
# # plt.imshow(predict)
# plt.subplot(1,2,1)
# plt.imshow(emb1[0])
# plt.subplot(1,2,2)
# plt.imshow(emb2[0])
# plt.show()

# # Computes distance between pairs
# diff = np.square(emb1-emb2)
# dist = np.sum(diff,1)
#
# best = 0
# best_t = 0
# thresholds = np.arange(0.001, 4, 0.001)
# for i in tqdm(range(len(thresholds))):
#     less = np.less(dist, thresholds[i])
#     acc = np.logical_not(np.logical_xor(less, issame))
#     acc = acc.astype(float)
#     out = np.sum(acc)
#     out = out/len(acc)
#     if out > best:
#         best_t = thresholds[i]
#         best = out
#
# print("-------------------------- Threshold / Accuracy --------------------------")
# print("Best threshold: " + str(best_t))
# print("Best accuracy: " + str(best))

# # Face Clustering
# kmeans = KMeans(n_clusters=len(np.unique(labels_test)), max_iter=2000, random_state=0, tol=0.2).fit(emb1)
#
# images_cluster = [images_test[np.equal(kmeans.labels_,i)] for i in range(len(labels_test))]
# labels_cluster = [labels_test[np.equal(kmeans.labels_,i)] for i in range(len(labels_test))]
#
# for i in range(len(images_cluster)):
#     length = len(images_cluster[i])
#     if length > 0:
#         print(labels_cluster[i])
#         fig = plt.figure(figsize=(length*2,2))
#         for j in range(length):
#             plt.subplot(1,length,j+1)
#             plt.imshow(images_cluster[i][j])
#             plt.xticks([])
#             plt.yticks([])
#         plt.show()