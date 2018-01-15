# -*- coding: utf-8 -*-
"""
Created on Mon Jan 01 18:17:28 2018

@author: carmelr
"""

#import os
#import argparse
#import sys
from dependency_parser.dependency_parser import DependencyParser, load_model

#project_dir = os.path.dirname(os.path.realpath('__file__'))
# project_dir = 'C:\\Users\\amirli\\Desktop\\amir\\NLP2'
project_dir = 'D:\\TECHNION\\NLP\\Dependency_Parser'
comp_path = project_dir + '\\data\\comp.unlabeled'
test_path = project_dir + '\\data\\test.labeled'
train_path = project_dir + '\\data\\train.labeled'
train_toy_path = project_dir + '\\data\\train_toy.labeled'
debug_path = project_dir + '\\data\\debug.labeled'

parser = DependencyParser()
parser.train(data_path=debug_path,
             test_path=debug_path,
             shuffle=False,
             patience=10,
             lr_patience=4,
             lr_factor=0.8,
             min_lr=0.1,
             init_w=None,
             max_iter=50,
             mode='complex')

# parser.train(data_path=train_path,
#              test_path=None,
#              shuffle=True,
#              max_iter=50,
#              mode='base')

parser.test(test_path)

results_path = project_dir + '\\results\\tmp'
parser.print_logs(results_path)

# parser.predict(test_path, results_path)

# parser.save_model(results_path)

# parser = load_model(results_path)
