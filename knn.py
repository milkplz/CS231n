import numpy as np
from cs231nlib.utils import load_CIFAR10

cifar10_dir = 'data/cifar-10-batches-py'
Xtr, Ytr, Xte, Yte = load_CIFAR10(cifar10_dir)

# Xtr.shape >>> (50000, 32, 32, 3)

# Xtr_rows.shape >>> (50000, 3072)
# Xte_rows.shape >>> (10000, 3072)

Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)
Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)

train_num = 5000
predict_num = 1000

Xval_rows = Xtr_rows[:predict_num, :]
Yval = Ytr[:predict_num]
Xtr_rows = Xtr_rows[predict_num:train_num+predict_num, :]
Ytr = Ytr[predict_num:train_num+predict_num]


# limi_train_num = 1000;
# limi_predict_num = 200;

# Xtr_rows = Xtr_rows[:limi_train_num, 0:Xtr_rows.shape[1]]
# Xte_rows = Xte_rows[:limi_predict_num, 0:Xte_rows.shape[1]]

# Ytr = Ytr[:limi_train_num]
# Yte = Yte[:limi_predict_num]

class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X, k=1):
        # 50000
        num_test = X.shape[0]
        # all value is 0, 50000x1 array
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        for i in xrange(num_test): # 0 ~ 10000 loop
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1) # distances 10000x1 res array, axis = 1 -> row
            tmp_label_predict = np.zeros(num_test, self.ytr.dtype)
            max_index = np.argmax(distances)
            max_distance = distances[max_index]
            # vote to labels
            for j in xrange(k):
                min_index = np.argmin(distances)
                label = self.ytr[min_index]
                # tmp_label_predict[label_index] = tmp_label_predict[label_index]+1
                tmp_label_predict[label] = tmp_label_predict[label]+(k+1-j)
                # print("var - ", j, k, (k+1-j), float((k+1-j)/(k+1)))
                distances[min_index] = max_distance

            max_index = np.argmax(tmp_label_predict)
            Ypred[i] = max_index

            # if i == 0:
                # print(tmp_label_predict, Ypred[i])
            
        return Ypred


# nn = NearestNeighbor()
# nn.train(Xtr_rows, Ytr)
# Yte_predict = nn.predict(Xte_rows, 5)
# print 'accuracy: %f' % ( np.mean(Yte_predict == Yte) )


for k in [1, 3, 5, 10, 20]:
    
    nn = NearestNeighbor()
    nn.train(Xtr_rows, Ytr)
    Yval_predict = nn.predict(Xval_rows, k = k)
    acc = np.mean(Yval_predict == Yval)
    print 'k : %d, accuracy : %f' % (k, acc)