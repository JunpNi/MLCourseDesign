import pickle
import numpy as np
import sys
import joblib
import gc


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict


def load_cifar10(n):
    data_train = []
    label_train = []
    # 融合训练集
    for i in range(1, 6):
        if i != n:
            print("载入gist_batch_%d为训练集" % i)
            dic = unpickle('gists_batch_' + str(i))
            for i_data in dic[b'gists']:
                data_train.append(i_data)
            for i_label in dic[b'labels']:
                label_train.append(i_label)
    # 融合验证集
    data_val = []
    label_val = []
    print("载入gist_batch_%d为测试" % n)
    dic = unpickle('gists_batch_' + str(n))
    for i_data in dic[b'gists']:
        data_val.append(i_data)
    for i_label in dic[b'labels']:
        label_val.append(i_label)
    return (np.array(data_train), np.array(label_train), np.array(data_val), np.array(label_val))


def load_cifarall():
    data_train = []
    label_train = []
    # 融合训练集
    for i in range(1, 6):
        print("载入gist_batch_%d为训练集" % i)
        dic = unpickle('gists_batch_' + str(i))
        for i_data in dic[b'gists']:
            data_train.append(i_data)
        for i_label in dic[b'labels']:
            label_train.append(i_label)
    # 融合验证集
    data_val = []
    label_val = []
    print("载入gist_test为测试")
    dic = unpickle('gists_test')
    for i_data in dic[b'gists']:
        data_val.append(i_data)
    for i_label in dic[b'labels']:
        label_val.append(i_label)
    return (np.array(data_train), np.array(label_train), np.array(data_val), np.array(label_val))


class NearestNeighbor():
    def __init__(self):
        pass

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict_label(self, dists, k):
        y_pred = np.zeros(dists.shape[0])
        for i in range(dists.shape[0]):
            # 取前K个标签
            closest_y = self.ytr[np.argsort(dists[i])[:k]]
            # 取K个标签中个数最多的标签
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred

    def predict(self, X):
        num_test = X.shape[0]
        self.X = X
        distances = np.zeros((X.shape[0], self.Xtr.shape[0]), dtype=self.Xtr.dtype)
        for i in range(num_test):
            distances[i]=np.sum(np.abs(self.Xtr-self.X[i]),axis=1)
            #distances[i] = np.sum(np.square(self.Xtr - self.X[i]), axis=1)
            if (i + 1) % 200 == 0:
                sys.stdout.write('\r')
                process = '[' + ((i + 1) // 200) * '#' + (50 - (i + 1) // 200) * '_' + ']' + str((i + 1) / 100) + '%'
                sys.stdout.write(process)
                sys.stdout.flush()
        return distances


def train(K):
    accuarcy = 0.00000
    dist = []
    for i in range(1, 6):
        allac = []
        (data_train, label_train, data_val, lable_val) = load_cifar10(i)
        print(data_val.shape[0])
        print(data_train.shape[0])
        distances = np.zeros((data_val.shape[0], data_train.shape[0]), dtype=data_val.dtype)
        nn = NearestNeighbor()
        nn.train(data_train, label_train)
        print("第%d轮dist计算开始" % i)
        distances = nn.predict(data_val)
        print("第%d轮dist计算完毕" % i)
        print("第%d轮预测开始" % i)
        for k in range(1, K + 1):
            # dist = joblib.load('gists_result_' + str(i))
            label_predict = np.zeros(data_val.shape, dtype=label_train.dtype)
            label_predict = nn.predict_label(distances, k)
            accuarcy = np.mean(lable_val == label_predict)
            # print("第%d轮准确率%f" %(i,np.mean(lable_val==label_predict)))
            allac.append(accuarcy)
            sys.stdout.write('\r')
            process = '[' + k // K * 100 * '#' + (100 - k // K * 100) * '_' + ']' + str(k / K) + '%'
            sys.stdout.write(process)
            sys.stdout.flush()
        fp = open('gist_result' + str(i), 'wb')
        pickle.dump(allac, fp, pickle.HIGHEST_PROTOCOL)
        fp.close()
        print("第%d轮预测完毕" % i)
        del distances
        gc.collect()
        # joblib.dump(distances, 'gists_result_' + str(i),0,pickle.HIGHEST_PROTOCOL)
    result = np.zeros(K)
    for i in range(1, 6):
        # dist = joblib.load('gists_result_' + str(i))
        temp = unpickle('gist_result' + str(i))
        print(temp)
        print(result)
        print(np.array(temp))
        result = result + np.array(temp)
        # print("第%d轮准确率%f" %(i,np.mean(lable_val==label_predict)))
    result = result / 5
    return result


def test(K):
    (data_train, label_train, data_val, lable_val) = load_cifarall()
    print(data_val.shape[0])
    print(data_train.shape[0])
    nn = NearestNeighbor()
    nn.train(data_train, label_train)
    print("dist计算开始")
    distances = nn.predict(data_val)
    print("dist计算完毕")
    print("预测开始")
    label_predict = nn.predict_label(distances, K)
    accuarcy = np.mean(lable_val == label_predict)
        # print("第%d轮准确率%f" %(i,np.mean(lable_val==label_predict)))
    del distances
    gc.collect()
    # joblib.dump(distances, 'gists_result_' + str(i),0,pickle.HIGHEST_PROTOCOL)
    # print("第%d轮准确率%f" %(i,np.mean(lable_val==label_predict)))
    return accuarcy

# result = train(20)
# for i in range(result.shape[0]):
#     print("k=%d,准确率为%f" % (i + 1, result[i]))
result=test(9)
print("k=%d,准确率为%f" % (9, result))
