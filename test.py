from supervised.util import Config, Experiment

from supervised.models.cnn import build_EfficientNetB0, build_hallucinetv4_upcycle_plus_plus

from supervised.datasets.image_classification import deep_weeds, cats_dogs
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
    'name': 'G1',
    'n_gpu': 1,
    'n_cpu': 12,
    'partition': 'ai2es',
    'nodelist': ['c732', 'c731'],
    'time': '48:00:00',
    'memory': 8196,
    'distributed': False,
    # The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
    'stdout_path': '/scratch/jroth/supercomputer/text_outputs/exp%01a_stdout_%A.txt',
    'stderr_path': '/scratch/jroth/supercomputer/text_outputs/exp%01a_stderr_%A.txt',
    'email': 'jay.c.rothenberger@ou.edu',
    'dir': '/scratch/jroth/AI2ES-DL/',
    'array': '[0]'
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
    'network_fn': build_hallucinetv4_upcycle_plus_plus,
    'network_args': {
        'lrate': 1e-3,
        'n_classes': 2,
        'iterations': 8,
        'conv_filters': '[16]',
        'conv_size': '[3]',
        'attention_heads': '[10]',
        'learning_rate': 1e-4,
        'image_size': (128, 128, 3),
        'l1': None,
        'l2': None,
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
    'patience': 12,
    'min_delta': 0.0,
    'epochs': 20,
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
    'dset_fn': cats_dogs,
    'dset_args': {
        'image_size': (128, 128, 3)
    },
    'cache': False,
    'batch': 4,
    'prefetch': 1,
    'shuffle': True,
    'augs': (mixup_dset,)
}

config = Config(hardware_params, network_params, dataset_params, experiment_params)

if __name__ == "__main__":

    print(config.experiment_params)

    exp = Experiment(config)

    print(exp.params)

    exp.enqueue()
