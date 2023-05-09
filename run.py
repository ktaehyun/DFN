from d_knn import lectureKNN
from dataPreprocessing import DataPreprocessing


temp = DataPreprocessing()
knn = lectureKNN()
user = temp.matrix
users = []
for u in user:
    users.append(u)


for i in users[:20]:
    tmp = knn.printClass(temp.matrix[i])
    exist_result = knn.existUser(i)
    print(f'Target-User => {i}', tmp)
    print(exist_result, '\n')

new = {'1895068':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 5, 0, 0, 0]}
new_result = knn.newUser(new)
print('New-User => 1895068', knn.printClass(new['1895068']))
print(new_result)