import os
from copy import deepcopy
import numpy as np
from .utils import load_data
from .model import Model
from datetime import datetime
import random
import torch


def init_seeds(seed=0, cuda_deterministic=True):
    print("Initializing seed...")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)  #if you are using multi-GPU
    if cuda_deterministic:
        #slower more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        #faster less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.enabled = False #this solution might increase the memory usage since it will disable cudnn for other modules like Cov2d.
 


def initialize_data(config, train=False, test=False):
    print("Initializing data source...")
    train_source, test_source = load_data(**config['data'], cache=(train or test))
    #_time1 = datetime.now()
    if train:
        print("Loading training data...")
        train_source.load_all_data()
    if test:
        print("Loading test data...")
        test_source.load_all_data()
    print("Data initialization complete.")
    #print(datetime.now() - _time1)
    return train_source, test_source


def initialize_model(config, train_source, test_source):
    print("Initializing model...")
    data_config = config['data']
    model_config = config['model']
    model_param = deepcopy(model_config)
    model_param['train_source'] = train_source
    model_param['test_source'] = test_source
    model_param['train_pid_num'] = data_config['pid_num']
    model_param['type_equalization'] = data_config['type_equalization']
    batch_size = int(np.prod(model_config['batch_size']))
    model_param['save_name'] = '_'.join(map(str, [
        model_config['model_name'],
        data_config['dataset'],
        data_config['pid_num'],
        data_config['pid_shuffle'],
        model_config['hidden_dim'],
        model_config['margin'],
        batch_size,
        model_config['hard_or_full_trip'],
        model_config['frame_num'],
    ]))

    model_param['teacher_save_name'] = '_'.join(map(str, [
        model_config['teacher_model_name'],
        data_config['dataset'],
        data_config['pid_num'],
        data_config['pid_shuffle'],
        model_config['teacher_hidden_dim'],
        model_config['margin'],
        batch_size,
        model_config['hard_or_full_trip'],
        model_config['frame_num'],
    ]))

    m = Model(**model_param)
    print("Model initialization complete.")
    return m, model_param['save_name']


def initialization(config,train=False,test=False):
    print("Initialzing...")
    WORK_PATH = config['WORK_PATH']
    os.chdir(WORK_PATH)
    init_seeds(seed=0, cuda_deterministic=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = config["CUDA_VISIBLE_DEVICES"]
    train_source, test_source = initialize_data(config, train, test)
    return initialize_model(config, train_source, test_source)