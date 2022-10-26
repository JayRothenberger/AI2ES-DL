from supervised.util import Config, Experiment

from supervised.models.cnn import build_EfficientNetB0

from supervised.datasets.image_classification import deep_weeds
from supervised.data_augmentation.msda import mixup_dset
from supervised.data_augmentation.ssda import add_gaussian_noise_dset
"""
hardware_params must include:

    'n_gpu': uint
    'n_cpu': uint
    'node': str
    'partition': str
    'time': str (we will just write this to the file)
    'memory': uint
    'distributed': bool
"""
hardware_params = {
    'n_gpu': 1,
    'n_cpu': 12,
    'node': 'str',
    'partition': 'str',
    'time': 'str',
    'memory': 'uint',
    'distributed': False
}
"""
network_params must include:
    
    'network_fn': network building function
    'network_args': arguments to pass to network building function
        network_args must include:
            'lrate': float
    'hyperband': bool
"""
network_params = {
    'network_fn': build_EfficientNetB0,
    'network_args': {
        'lrate': 1e-3,
        'n_classes': 9,
    },
    'hyperband': False
}
"""
experiment_params must include:
    
    'seed': random seed for computation
    'steps_per_epoch': uint
    'validation_steps': uint
    'patience': uint
    'min_delta': float
    'epochs': uint
    'nogo': bool
"""
experiment_params = {
    'seed': 42,
    'steps_per_epoch': 100,
    'validation_steps': 100,
    'patience': 1,
    'min_delta': 0.0,
    'epochs': 2,
    'nogo': False,
}
"""
dataset_params must include:
    'dset_fn': dataset loading function
    'dset_args': arguments for dataset loading function
    'cache': str or bool
    'batch': uint
    'prefetch': uint
    'shuffle': bool
    'augs': list of augs
"""
dataset_params = {
    'dset_fn': deep_weeds,
    'dset_args': {
        'image_size': (256, 256, 3)
    },
    'cache': False,
    'batch': 1,
    'prefetch': 1,
    'shuffle': True,
    'augs': [add_gaussian_noise_dset, mixup_dset]
}

config = Config(hardware_params, network_params, dataset_params, experiment_params)

print(config.experiment_params)

exp = Experiment(config)

print(exp.params)

exp.run()
