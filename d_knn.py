import numpy as np
from copy import deepcopy
from dataPreprocessing import DataPreprocessing




class lectureKNN(DataPreprocessing):


    def cosineSimillarity(self, v1, v2):
        A = np.sqrt(np.sum(np.square(v1)))
        B = np.sqrt(np.sum(np.square(v2)))

        return np.dot(v1,v2) / (A*B)


    def recommenderLecture(self, target_uid, lecture):
        lst = [0 for _ in range(12)]
        prio, mid, mino, els = lst, lst, lst, lst
        cnt, result = 0, []
        target_lecture = self.matrix[target_uid][:12]
        for lect in lecture:
            for idx, lec in enumerate(lect):
                if target_lecture[idx]==0 and lec==10:
                    prio[idx] += 1
                elif target_lecture[idx]==0 and (lec==5 or lec==4 or lec==3):
                    mid[idx] += 1
                elif lec>=2 and target_lecture[idx]<lec:
                    mino[idx] += 1
                elif target_lecture[idx] < lec:
                    els[idx] += 1
        del target_lecture, lecture

        for idx in range(12):  # prio, mid, mino, els
            try:
                prio_tmp, mid_tmp, mino_tmp, els_tmp = [], [], [], []
                if prio[idx] != 0:
                    prio_tmp.append((idx, prio[idx]))
                elif mid[idx] != 0:
                    mid_tmp.append((idx, mid[idx]))
                elif mino[idx] != 0:
                    mino_tmp.append((idx, mino[idx]))
                elif els[idx] != 0:
                    els_tmp.append((idx, els[idx]))

                prio_tmp.sort(key=lambda x: x[1], reverse=True)
                mid_tmp.sort(key=lambda x: x[1], reverse=True)
                mino_tmp.sort(key=lambda x: x[1], reverse=True)
                els_tmp.sort(key=lambda x: x[1], reverse=True)
                tmp = prio_tmp + mid_tmp + mino_tmp + els_tmp

                for i, v in tmp:
                    if i not in result:
                        result.append(self.lectures[i])
                        cnt += 1
                        if cnt == 3:
                            return result
            except:
                continue

        return result


    def preprocessingSimilarity(self, uid, lst):
        cnt, lecture_lst = 0, []
        for l in lst:
            if l[1] >= 0.6:
                cnt += 1
                user_uid = l[0]
                lecture_lst.append(self.matrix[user_uid][:12])
                if cnt == 3:
                    return self.recommenderLecture(uid, lecture_lst)
            else:
                return self.recommenderLecture(uid, lecture_lst)


    def printClass(self, user_lst):
        tmp = []
        for i, t in enumerate(user_lst[-6:]):
            if t != 0:
                if i == 0:
                    tmp.append('공학')
                elif i == 1:
                    tmp.append('인문/사회')
                elif i == 2:
                    tmp.append('예체능')
                elif i == 3:
                    tmp.append('교육')
                elif i == 4:
                    tmp.append('자연')
                elif i == 5:
                    tmp.append('의약')

        return tmp


    def newUser(self, new_user):
        new_temp = []
        for k, v in zip(new_user.keys(), new_user.values()):
            new_uid, new_vec = k, v
        for idx, vec in zip(self.matrix.keys(), self.matrix.values()):
            similarity = self.cosineSimillarity(vec, new_vec)
            new_temp.append((idx, similarity))
        new_temp.sort(key=lambda x: x[1], reverse=True)
        self.matrix[new_uid] = new_vec
        print('\n', new_temp[:3])
        for i in range(3):
            print(f'{new_temp[i][0]}', self.printClass(self.matrix[new_temp[i][0]]))
        return self.preprocessingSimilarity(new_uid, new_temp)


    def existUser(self, user_uid):
        exist_temp = []
        for idx, vec in zip(self.matrix.keys(), self.matrix.values()):
            if idx != user_uid:
                similarity = self.cosineSimillarity(vec, self.matrix[user_uid])
                exist_temp.append((idx, similarity))
        exist_temp.sort(key=lambda x: x[1], reverse=True)
        print(exist_temp[:3])
        for i in range(3):
            print(f'{exist_temp[i][0]}', self.printClass(self.matrix[exist_temp[i][0]]))
        return self.preprocessingSimilarity(user_uid, exist_temp)