import numpy as np
import skimage as sk
from contact import CSMLR

faceId = CSMLR()

SIZE = (224,224,3)
h, w, c = SIZE
image = np.empty((1, h, w, c))
image[0] = sk.io.imread('C:/Users/ktaehyun/PycharmProjects/DogFaceNet/data/dogfacenet/aligned/after_4_bis/0/0.0.jpg')
image /= 255.0
result = faceId.existUser({150.0:image})
print(result)
