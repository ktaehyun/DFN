import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K


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

model = tf.keras.models.load_model('C:/Users/ktaehyun/PycharmProjects/DogFaceNet/output/model/2023.04.29.test/dogfacenet.test.3.h5', custom_objects={'triplet':triplet,'triplet_acc':triplet_acc})

class CSMLR():
    def __init__(self):
        data = pd.read_csv('./test.csv')
        self.matrix = {}
        for i in range(len(data)):
            row = data.iloc[i]
            try:
                self.matrix[row[0]].append(list(row[1:]))
            except:
                self.matrix[row[0]] = [list(row[1:])]

    def cosineSimillarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    # def newUser(self, new_user):
    #     new_temp = []
    #     for k, v in zip(new_user.keys(), new_user.values()):
    #         new_uid, new_vec = k, v
    #     for idx, vec in zip(self.matrix.keys(), self.matrix.values()):
    #         similarity = self.cosineSimillarity(vec, new_vec)
    #         new_temp.append((idx, similarity))
    #     new_temp.sort(key=lambda x: x[1], reverse=True)
    #     self.matrix[new_uid] = new_vec
    #     print('\n', new_temp[:3])
    #     for i in range(3):
    #         print(f'{new_temp[i][0]}', self.printClass(self.matrix[new_temp[i][0]]))
    #     return self.preprocessingSimilarity(new_uid, new_temp)

    def existUser(self, dog_info):
        exist_temp, tmp = [], []
        # 강아지 ID, Image 처리
        for idx, img in zip(dog_info.keys(), dog_info.values()):
            dog_id, dog_picture = idx, img
            dog_vec = model.predict(dog_picture)
        # 유사도 계산
        for idx, vec in zip(self.matrix.keys(), self.matrix.values()):
            for v in vec:
                similarity = self.cosineSimillarity(v, list(dog_vec[0]))
                exist_temp.append((idx, similarity))
        exist_temp.sort(key=lambda x: x[1], reverse=True)
        print(exist_temp[:5])
        for i in range(5):
            print(f'{exist_temp[i][0]}')
            tmp.append(exist_temp[i][0])
        return tmp