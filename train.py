from datetime import datetime
from model.initialization import initialization
from config import conf
import argparse


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--cache', default=True, type=boolean_string,
                    help='cache: if set as TRUE all the training data will be loaded at once'
                         ' before the training start. Default: TRUE')
parser.add_argument('--distillation', default=True, type=boolean_string,
                    help='distillation: if set as TRUE the teacher model need be loaded'
                         ' before the training start. Default: TRUE')
parser.add_argument('--teacher_model_iter', default='80000', type=int,
                    help='teacher_model_iter: iteration of the checkpoint of teacher model to load. Default: 80000')


opt = parser.parse_args()
m = initialization(conf, train=opt.cache)[0]
if opt.distillation:
    print('Loading the teacher model of iteration %d...' % opt.teacher_model_iter)
    m.load_teacher_model(opt.teacher_model_iter)
print("Training START")
m.fit()
print("Training COMPLETE")
