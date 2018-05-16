# loading library
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=100)

'''
X_train = X_trains
X_train.extend(X_traint)
y_train = y_trains
y_train.extend(y_traint)

ss = []
tt = []
'''
# fitting the model

source = np.load('source.npy')
target = np.load('target.npy')
mixed = np.load('mixed.npy')

print source[0]

oi = []
for i in range(1000):
   oi.append(1)  # target
for i in range(1000):
   oi.append(0)  # source   

ans = 0

#print 'mixed[0] ', len(mixed)
print len(mixed[0])
knn.fit(mixed, oi)

for i in range(1000):
    op = source[i].reshape(1, 9728)
    pred = knn.predict(op)
    #print pred
    #exit()
    if( pred == [1] ):
         ans += 1

ans = float(ans)/len(source)

print ans

'''
for i in range(1000):
    op = target[i].reshape(1, 9728)
    pred = knn.predict(op)
    print pred
    if( pred == [0] ):
         ans += 1

ans = float(ans)/len(source)

print ans
'''