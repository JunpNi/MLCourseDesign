import pickle
import numpy as np
from PIL import Image
import leargist


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict


def load_cifar10():
    data_train = []
    label_train = []
    # 融合训练集
    dic = unpickle('test_batch')
    pic = dic[b'data'].reshape(10000, 3, 32, 32)
    gist = []
    for img in pic:
        i0 = Image.fromarray(img[0])  # 从数据，生成image对象
        i1 = Image.fromarray(img[1])
        i2 = Image.fromarray(img[2])
        img = Image.merge("RGB", (i0, i1, i2))
        des = leargist.color_gist(img)
        gist.append(des)
    newdic = {b'gists': gist, b'labels': dic[b'labels']}
    print("测试计算完毕")
    fp = open('gists_test', 'wb')
    pickle.dump(newdic, fp)
    fp.close()
    for i in range(1, 6):
        dic = unpickle('data_batch_' + str(i))
        pic = dic[b'data'].reshape(10000, 3, 32, 32)
        gist=[]
        for img in pic:
            i0 = Image.fromarray(img[0])  # 从数据，生成image对象
            i1 = Image.fromarray(img[1])
            i2 = Image.fromarray(img[2])
            img = Image.merge("RGB", (i0, i1, i2))
            des = leargist.color_gist(img)
            gist.append(des)
        newdic={b'gists':gist,b'labels':dic[b'labels']}
        print("第%d轮计算完毕"%i)
        fp = open('gists_batch_' + str(i), 'wb')
        pickle.dump(newdic,fp)
        fp.close()

        # print(dic[b'gists'])
    # for i_label in dic[b'labels']:
    #     label_train.append(i_label)


load_cifar10()
