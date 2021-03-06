conf = {
    "WORK_PATH": "./work-distilled",
    "CUDA_VISIBLE_DEVICES": "0",
    "data": {
        'dataset_path': "your_path",
        'resolution': '64',
        'dataset': 'CASIA-B',
        'pid_num': 73,
        'pid_shuffle': False,
        'type_equalization': True,
    },
    "model": {
        'distillation': True,
        'distillation_weight': 1.0,
        'hidden_dim': 128,
        'teacher_hidden_dim':256,
        'lr': 1e-1,
        'lr_decay_rate':0.1,
        'optimizer_type': 'sgd',
        'momentum':0.9,
        'weight_decay':0.0, #5e-4
        'hard_or_full_trip': 'full',
        'batch_size': (8, 16),
        'restore_iter': 0,
        'total_iter': 80000,
        'margin': 0.2,
        'num_workers': 4,
        'frame_num': 30,
        'model_name': 'LightGaitSet',
        'teacher_model_name':'GaitSet',
    },
}

