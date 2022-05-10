from datetime import datetime
import numpy as np
import argparse

from model.initialization import initialization
from model.utils import evaluation
from config import conf
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--iter', default='80000', type=int,
                    help='iter: iteration of the checkpoint to load. Default: 0')
parser.add_argument('--batch_size', default='1', type=int,
                    help='batch_size: batch size for parallel test. Default: 1')
parser.add_argument('--cache', default=False, type=boolean_string,
                    help='cache: if set as TRUE all the test data will be loaded at once'
                         ' before the transforming start. Default: FALSE')
opt = parser.parse_args()


def de_diag(acc, each_angle=False):
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / 10.0
    if not each_angle:
        result = np.mean(result)
    return result


m = initialization(conf, test=opt.cache)[0]
print('Loading the model of iteration %d...')
m.load(opt.iter)
print('Transforming...')
time = datetime.now()
test = m.transform('test', opt.batch_size)
print('Evaluating...')
acc = evaluation(test, conf['data'])
print('Evaluation complete. Cost:', datetime.now() - time)
for k in range(1):
    print('===Rank-%d (Exclude identical-view cases)===' % (k + 1))
    print('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
        de_diag(acc[0, :, :, k]),
        de_diag(acc[1, :, :, k]),
        de_diag(acc[2, :, :, k])))
    
    
np.set_printoptions(precision=2, floatmode='fixed')
for i in range(1):
    print('===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
    print('NM:', de_diag(acc[0, :, :, i], True))
    print('BG:', de_diag(acc[1, :, :, i], True))
    print('CL:', de_diag(acc[2, :, :, i], True))
    
        




