import numpy as np
import tensorflow as tf
from copy import deepcopy
from dataPreprocessing import DataPreprocessing


model = tf.keras.models.load_model('C:/Users/ktaehyun/PycharmProjects/DogFaceNet/output/model/2023.04.29.test/dogfacenet.test.3.h5', custom_objects={'triplet':triplet,'triplet_acc':triplet_acc})

class CSMLR(DataPreprocessing):

    def cosineSimillarity(self, v1, v2):
        A = np.sqrt(np.sum(np.square(v1)))
        B = np.sqrt(np.sum(np.square(v2)))

        return np.dot(v1,v2) / (A*B)

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
            similarity = self.cosineSimillarity(vec, dog_vec)
            exist_temp.append((idx, similarity))
        exist_temp.sort(key=lambda x: x[1], reverse=True)
        print(exist_temp[:4])
        for i in range(4):
            print(f'{exist_temp[i][0]}')
            tmp.append(exist_temp[i][0])
        return tmp