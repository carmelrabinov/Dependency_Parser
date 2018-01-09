# -*- coding: utf-8 -*-
"""
Created on Mon Jan 01 18:17:28 2018

@author: carmelr
"""

#import os
#import argparse
#import sys
#from dependency_parser.dependency_parser import data_preprocessing
import time
# from dependency_parser.chu_liu import Digraph
from dependency_parser import dependency_parser

#project_dir = os.path.dirname(os.path.realpath('__file__'))
# project_dir = 'C:\\Users\\amirli\\Desktop\\amir\\NLP2'
project_dir = 'D:\\TECHNION\\NLP\\Dependency_Parser'
comp_path = project_dir + '\\data\\comp.unlabeled'
test_path = project_dir + '\\data\\test.labeled'
train_path = project_dir + '\\data\\train.labeled'
debug_path = project_dir + '\\data\\debug.labeled'

parser = dependency_parser.DependencyParser()
parser.train(train_path, max_iter=20)
parser.save_model('results\\base_features_all_train_set')

accuracy = parser.test(test_path)
print('accuracy over test set is ', accuracy)
accuracy = parser.test(test_path)
print('accuracy over test set is ', accuracy)
accuracy = parser.test(test_path)
print('accuracy over test set is ', accuracy)
accuracy = parser.test(test_path)
print('accuracy over test set is ', accuracy)

