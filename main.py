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
results_path = 'results\\complex_gap_features'

parser = DependencyParser()
parser.train(train_toy_path, max_iter=20, mode='complex')

# parser.save_model(results_path)

# parser = load_model(results_path)
for _ in range(5):
    parser.test(test_path)

parser.print_logs(results_path)
