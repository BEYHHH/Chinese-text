#-*-coding:utf-8-*-
import jieba
import os
import jieba.analyse
from optparse import OptionParser
import sys
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn import neighbors, datasets
import sklearn.feature_extraction
from sklearn.neighbors import KNeighborsClassifier
import random
from sklearn import metrics

reload(sys)
sys.setdefaultencoding('utf-8')
def get_list(path):
    f = open("Reduced/"+ path, 'r+')
    t = f.read()
    f.close()
    try:
        text = t.decode('gbk')
        seg_list = jieba.cut(text, cut_all=False)
        result = []
    except BaseException:
        os.remove("Reduced/"+ path)
        return
    for seg in seg_list :
        seg = ''.join(seg.split())
        if (seg != '' and seg != "\n" and seg != "\n\n"):
            result.append(seg)
    f = open("ready/Reduced/"+path,"w+")
    f.write(' '.join(result))
    f.close()

if __name__ == '__main__':
    n = random.randint(0,8)
    path = "Reduced"
    files = os.listdir(path)
    The_order = 0
    for file_name in files:
        print file_name+ "begine"
        for name in  os.listdir(path +'/'+file_name):
            get_list(file_name + '/' +name)
        print "the data is gotten" + file_name