# -*- coding: utf-8 -*-
"""
Created on Mon Jan 01 18:17:28 2018

@author: carmelr
"""

import os
import argparse
import sys
from dependency_parser.dependency_parser import DependencyParser, load_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('results_dir', help='output results directory')
    parser.add_argument('-train_path', type=str, default=None)
    parser.add_argument('-comp_path', type=str, default=None)
    parser.add_argument('-test_path', type=str, default=None)
    parser.add_argument('-max_iter', type=int, default=50)
    parser.add_argument('-mode', type=str, default='complex')
    parser.parse_args(namespace=sys.modules['__main__'])

    project_dir = os.path.dirname(os.path.realpath('__file__'))
    if comp_path is None:
        comp_path = project_dir + '\\data\\comp.unlabeled'
    if test_path is None:
        test_path = project_dir + '\\data\\test.labeled'
    if train_path is None:
        train_path = project_dir + '\\data\\train.labeled'

    results_path = project_dir + '\\results\\' + str(results_dir)

    parser = DependencyParser()
    parser.train(data_path=train_path,
                 test_path=test_path,
                 shuffle=True,
                 patience=4,
                 lr_patience=4,
                 lr_factor=0.3,
                 min_lr=0.02,
                 bucketing=True,
                 max_iter=max_iter,
                 mode=mode)
    parser.print_logs(results_path)
    # parser.save_weights(results_path)
    # parser.save_model(results_path)
    parser.predict(comp_path, results_path)

    # parser.analysis(test_path, results_path)
    # parser = load_model(results_path)

