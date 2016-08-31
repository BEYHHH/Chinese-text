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
num = 0
dic = {}

fl=open('result/test_with_KNN_100_5.txt', 'a')

def calculate_result(actual,pred):
    print "pred"
    print pred
    print len(pred)
    """
    for a in range(len(pred)):
        print str(pred[a]) + "   the tage is "+ str(actual[a])
    """
    m_precision = metrics.precision_score(actual , pred, average = 'weighted')
    m_recall = metrics.recall_score(actual, pred, average = 'weighted')
    fl.write('\npredict info:')
    fl.write('\nprecision:{0:.3f}'.format(m_precision))
    fl.write('\nrecall:{0:0.3f}'.format(m_recall))
    fl.write('\nf1-score:{0:.3f}'.format(metrics.f1_score(actual,pred, average = 'weighted')))
             
def get_list(path):
    f= open(path,'r+')
    t = f.read()
    f.close()
    return t




   
    
def test_paramet( X_train, Y_train, X_test, Y_test, max_df, min_df, Neigbours_number):
    fl.write("\n\n\nmax_df: " + str(max_df))
    fl.write("\nmin_df: " + str(min_df))
    fl.write("\nKNN: " + str(Neigbours_number))
    tv1 = TfidfVectorizer( max_df = max_df , min_df = min_df)
    knnclf = KNeighborsClassifier(weight = "distance",n_neighbors = Neigbours_number)
    tfidf_train = tv1.fit_transform(X_train)
    knnclf.fit( tfidf_train, Y_train) 
    tv2 = TfidfVectorizer(vocabulary = tv1.vocabulary_)
    tfidf_test = tv2.fit_transform(X_test)
    pred = knnclf.predict(tfidf_test)
    
    fl.write( "\n the shape of train is "+ repr(tfidf_train.shape))
    fl.write( "\n the shape of test is "+ repr(tfidf_test.shape))
    
    calculate_result(Y_test,pred)
    
    
if __name__ == '__main__':
    n = random.randint(0,8)
    path = "ready/Reduced"
    files = os.listdir(path)
    The_order = 0
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    for a in files:
        print a
    print len(files)
    for file_name in files:
        print file_name + "   begine"
        if file_name == ".ipynb_checkpoints":
            continue
        dic[The_order] = file_name
        The_order += 1
        train_num = 0
        test_num = 0
        for name in  os.listdir(path +'/'+file_name):
            text_number = int(name[:-4])
            if text_number > 1600:
                test_X.append(get_list(path + '/' + file_name + '/' +name))
                test_num += 1
            else:
                train_X.append(get_list(path + '/' + file_name + '/' +name))
                train_num += 1
        train_Y += [The_order - 1] * train_num
        test_Y +=  [The_order - 1] * test_num
        print "the data is gotten   " + file_name
    for Max in range(1000,4001,100):
        for Min in range(100,301,5):
            test_paramet( train_X, train_Y, test_X, test_Y, Max, Min, 5)