import os
import pickle
from time import time, sleep
from random import randint
import json

import numpy.random
import tensorflow as tf
from supervised.data_structures import ModelData

import pynvml
import numpy as np


class Config:
    hardware_params = None
    network_params = None
    dataset_params = None
    experiment_params = None

    def __init__(self, hardware_params=None, network_params=None, dataset_params=None, experiment_params=None):
        self.hardware_params = hardware_params
        self.network_params = network_params
        self.dataset_params = dataset_params
        self.experiment_params = experiment_params

    def dump(self, fp):
        pickle.dump(self, fp)

    def load(self, fp):
        obj = pickle.load(fp)
        self.hardware_params = obj.hardware_params
        self.network_params = obj.network_params
        self.dataset_params = obj.dataset_params
        self.experiment_params = obj.experiment_params


def prep_gpu(cpus_per_task, gpus_per_task=0, wait=True):
    """prepare the GPU for tensorflow computation"""
    # initialize the nvidia management library
    pynvml.nvmlInit()
    # if we are not to use the gpu, then disable them
    if not gpus_per_task:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        n_physical_devices = 0
    else:
        # gpu handles
        gpus = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        # -1 means "use every gpu"
        gpus_per_task = gpus_per_task if gpus_per_task > -1 else len(gpus)
        # get the fraction of used memory for each gpu
        if wait:
            sleep(randint(0, 60 * 5))
        usage = [pynvml.nvmlDeviceGetMemoryInfo(gpu).used / pynvml.nvmlDeviceGetMemoryInfo(gpu).total for gpu in gpus]
        # sort the gpus by their available memory and filter out all gpus with more than 10% used
        avail = [i for i, v in sorted(list(enumerate(usage)), key=lambda k: k[-1], reverse=False) if v <= 1]
        # if we cannot satisfy the requested number of gpus this is an error
        if gpus_per_task > len(gpus):
            raise ValueError("too many gpus requested for this machine")
        # get the physical devices from tensorflow
        physical_devices = tf.config.get_visible_devices('GPU')
        # only take the number of gpus we plan to use
        avail_physical_devices = [physical_devices[i] for i in avail][:gpus_per_task]
        # set the visible devices only to the <gpus_per_task> least utilized
        tf.config.set_visible_devices(avail_physical_devices, 'GPU')
        n_physical_devices = len(avail_physical_devices)

    # use the available cpus to set the parallelism level
    if cpus_per_task is not None:
        pass
        #tf.config.threading.set_inter_op_parallelism_threads(cpus_per_task)
        #tf.config.threading.set_intra_op_parallelism_threads(cpus_per_task)

    if n_physical_devices > 1:
        for physical_device in physical_devices:
            tf.config.experimental.set_memory_growth(physical_device, True)
        print('We have %d GPUs\n' % n_physical_devices)
    elif n_physical_devices:
        print('We have %d GPUs\n' % n_physical_devices)
    else:
        print('NO GPU')


def generate_fname(args):
    # TODO: fix this
    """
    Generate the base file name for output files/directories.

    The approach is to encode the key experimental parameters in the file name.  This
    way, they are unique and easy to identify after the fact.

    :param args: from argParse
    :params_str: String generated by the JobIterator
    :return: a string (file name prefix)
    """
    return f"{str(time()).replace('.', '')[-6:]}"


def execute_exp(model, train_dset, val_dset, network_params, experiment_params,
                train_steps=None, val_steps=None, callbacks=None, evaluate_on=None):
    """
    Perform the training and evaluation for a single model

    :param args: Argparse arguments object
    :param model: keras model
    :param train_dset: training dataset instance
    :param val_dset: validation dataset instance
    :param train_iteration: training iteration
    :param train_steps: number of training steps to perform per epoch
    :param val_steps: number of validation steps to perform per epoch
    :param callbacks: callbacks for model.fit
    :param evaluate_on: a dictionary of objects on which to call model.evaluate (take care they are not infinite)
    :return: trained keras model encoded as a ModelData instance
    """
    start = time()
    print(model.summary())
    # tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, expand_nested=True,
    #                          to_file=os.curdir + f'/../visualizations/models/model_{str(time())[:6]}.png')
    # Perform the experiment?
    if experiment_params['nogo']:
        # No!
        print("NO GO")
        return

    # Learn
    #  steps_per_epoch: how many batches from the training set do we use for training in one epoch?
    #  validation_steps=None
    #  means that ALL validation samples will be used (of the selected subset)
    print(val_steps)
    history = model.fit(train_dset,
                        epochs=experiment_params['epochs'],
                        verbose=True,
                        validation_data=val_dset,
                        callbacks=callbacks,
                        steps_per_epoch=train_steps,
                        shuffle=False)

    model.save('../results/stupid_models/' + ''.join(str(time()).split('.')))

    evaluate_on = dict() if evaluate_on is None else evaluate_on

    evaluations = {k: model.evaluate(evaluate_on[k], steps=val_steps) for k in evaluate_on}
    end = time()
    # populate results data structure
    model_data = ModelData(weights=model.get_weights(),
                           network_params=network_params,
                           network_fn=network_params['network_fn'],
                           evaluations=evaluations,
                           classes=network_params['network_args']['n_classes'],
                           history=history.history,
                           run_time=end - start)
    print('returning model data')

    return model_data


def start_training(model,
                   train_dset,
                   val_dset,
                   network_params,
                   experiment_params,
                   evaluate_on=None,
                   train_steps=None,
                   val_steps=None):
    """
    train a keras model on a dataset and evaluate it on other datasets then return the ModelData instance

    :param model: keras model
    :param train_dset: tf.data.Dataset for training
    :param val_dset: tf.data.Dataset for evaluation
    :param evaluate_on: a dictionary of finite objects passable to model.evaluate
    :param train_steps: number of steps per epoch
    :param val_steps: number of validations steps per epoch
    :return: a ModelData instance
    """
    # Override arguments if we are using exp_index

    train_steps = train_steps if train_steps is not None else 100

    print(train_steps, val_steps)

    evaluate_on = dict() if evaluate_on is None else evaluate_on

    def bleed_out(index, lrate=1e-3, minimum=1e-3):
        # Oscillating learning rate schedule
        # inspired by https://arxiv.org/abs/1506.01186
        from math import sin, pi
        x = index + 1
        frac = (1 - minimum) / x ** (1 - sin(2 * pi * (x ** .5)))
        return min(network_params['network_args']['lrate'] * (frac + minimum), 1)

    def cyclical_adv_lrscheduler25(epoch, lrate):
        """CAI Cyclical and Advanced Learning Rate Scheduler.
        # Arguments
            epoch: integer with current epoch count.
        # Returns
            float with desired learning rate.
        """
        base_learning = 0.001
        local_epoch = epoch % 25
        if local_epoch < 7:
            return base_learning * (1 + 0.5 * local_epoch)
        else:
            return (base_learning * 4) * (0.85 ** (local_epoch - 7))

    def loss_weight_schedule(epoch):
        """
        loss weight schedule for clam model
        """
        return 0 if epoch < 3 else 1

    #from supervised.models.cnn import LossWeightScheduler

    callbacks = [
                 tf.keras.callbacks.EarlyStopping(patience=experiment_params['patience'],
                                                  restore_best_weights=True,
                                                  min_delta=experiment_params['min_delta'],
                                                  monitor='val_clam_categorical_accuracy'),
                 tf.keras.callbacks.LearningRateScheduler(bleed_out),
                 #LossWeightScheduler(loss_weight_schedule)
                 ]

    return execute_exp(model, train_dset, val_dset, network_params, experiment_params,
                       train_steps, callbacks=callbacks, evaluate_on=evaluate_on)


from itertools import product


class JobIterator():
    def __init__(self, params):
        """
        Constructor

        @param params Dictionary of key/list pairs
        """
        self.params = params
        # List of all combinations of parameter values
        self.product = list(dict(zip(params, x)) for x in product(*params.values()))
        # Iterator over the combinations
        self.iter = (dict(zip(params, x)) for x in product(*params.values()))

    def next(self):
        """
        @return The next combination in the list
        """
        return self.iter.next()

    def get_index(self, i):
        """
        Return the ith combination of parameters

        @param i Index into the Cartesian product list
        @return The ith combination of parameters
        """

        return self.product[i]

    def get_njobs(self):
        """
        @return The total number of combinations
        """

        return len(self.product)

    def set_attributes_by_index(self, i, obj):
        """
        For an arbitrary object, set the attributes to match the ith job parameters

        @param i Index into the Cartesian product list
        @param obj Arbitrary object (to be modified)
        @return A string representing the combinations of parameters, and the args object
        """

        # Fetch the ith combination of parameter values
        d = self.get_index(i)
        # Iterate over the parameters
        for k, v in d.items():
            obj[k] = v

        return obj, self.get_param_str(i)

    def get_param_str(self, i):
        """
        Return the string that describes the ith job parameters.
        Useful for generating file names

        @param i Index into the Cartesian product list
        """

        out = 'JI_'
        # Fetch the ith combination of parameter values
        d = self.get_index(i)
        # Iterate over the parameters
        for k, v in d.items():
            out = out + "%s_%s_" % (k, v)

        # Return all but the last character
        return out[:-1]


def augment_args(index, network_params):
    """
    Use the jobiterator to override the specified arguments based on the experiment index.

    Modifies the args

    :param args: arguments from ArgumentParser
    :return: A string representing the selection of parameters to be used in the file name
    """

    # Create parameter sets to execute the experiment on.  This defines the Cartesian product
    #  of experiments that we will be executing
    p = network_params['network_args']
    p['network_fn'] = network_params['network_fn']

    for key in p:
        if not isinstance(p[key], list):
            p[key] = [p[key]]

    # Check index number
    if index is None:
        return ""
    # Create the iterator
    ji = JobIterator({key: p[key] for key in set(p) - set(p['search_space']) - {'search_space'}}) \
        if network_params['hyperband'] else JobIterator({key: p[key] for key in set(p) - {'search_space'}})

    print("Size of Hyperparameter Grid:", ji.get_njobs())

    # Check bounds
    assert (0 <= index < ji.get_njobs()), "exp out of range"

    # Push the attributes to the args object and return a string that describes these structures
    augmented, arg_str = ji.set_attributes_by_index(index, network_params)

    if network_params['hyperband']:
        vars(augmented).update({key: p[key] for key in p['search_space']})
        vars(augmented).update({'search_space': p['search_space']})

    augmented = {
        'network_fn': augmented['network_fn'],
        'network_args': augmented,
        'hyperband': network_params['hyperband']
    }
    augmented['network_args'].pop('network_fn')
    augmented['network_args'].pop('network_args')

    return augmented, arg_str


def dict_to_string(d: dict, prefix="\t"):
    # print each key/value pair from the dict on a new line
    s = "{\n"
    for k in d:
        if isinstance(d[k], dict):
            newfix = prefix + '\t'
            s += f"{prefix}{k}: {dict_to_string(d[k], newfix)}\n"
        else:
            s += f"{prefix}{k}: {d[k]}\n"
    return s + prefix + "}"


class Experiment:
    """
        The Experiment class is used to run and enqueue deep learning jobs with a config dict
    """

    def __init__(self, config):
        self.index = None
        self.batch_file = None
        self.network_params = config.network_params
        self.hardware_params = config.hardware_params
        self.dataset_params = config.dataset_params
        self.params = config.experiment_params
        self.run_args = None

    def run(self):
        """
        1. prep the hardware (prep_gpu)
        2. get the data
        3. make the model
        4. fit the model
        5. save the data
        """
        print(dict_to_string(self.hardware_params))
        print(dict_to_string(self.params))
        print(dict_to_string(self.network_params))
        print(dict_to_string(self.dataset_params))
        # set seed
        tf.random.set_seed(self.params['seed'])
        numpy.random.seed(self.params['seed'])
        try:
            prep_gpu(self.hardware_params['n_cpu'], self.hardware_params['n_gpu'], False)
        except:
            pass

        network_fn = self.network_params['network_fn']
        network_args = self.network_params['network_args']

        if self.hardware_params['n_gpu'] > 1:
            # create the scope
            strategy = tf.distribute.MirroredStrategy()
            with strategy.scope():
                # build the model (in the scope)
                model = network_fn(**network_args)
        else:
            model = network_fn(**network_args)

        dset_dict = self.dataset_params['dset_fn'](**self.dataset_params['dset_args'])

        train_dset, val_dset, test_dset = dset_dict['train'], dset_dict['val'], dset_dict['test']

        def postprocess_dset(ds):
            ds = ds.repeat()

            if self.dataset_params['batch'] > 1:
                ds = ds.batch(self.dataset_params['batch'], drop_remainder=True)

            if self.dataset_params['cache']:
                if self.dataset_params['cache_to_lscratch']:
                    ds = ds.cache(self.run_args.lscratch)
                else:
                    ds = ds.cache()

            if self.dataset_params['shuffle']:
                ds = ds.shuffle(self.dataset_params['shuffle'], self.params['seed'], True)

            ds = ds.prefetch(self.dataset_params['prefetch'])

            return ds

        val_dset = val_dset.batch(self.dataset_params['batch'])

        train_dset = postprocess_dset(train_dset)

        for aug in self.dataset_params['augs']:
            train_dset = aug(train_dset)
        # train the model
        if self.network_params['hyperband']:
            return start_training(model, train_dset, val_dset, self.network_params, self.params,
                                  train_steps=self.params['steps_per_epoch'],
                                  evaluate_on={'test': test_dset})

        model_data = start_training(model, train_dset, val_dset, self.network_params, self.params,
                                    train_steps=self.params['steps_per_epoch'])

        result = Results(self, model_data)
        with open(f'{os.curdir}/../results/{generate_fname(self.params)}', 'wb') as fp:
            pickle.dump(result, fp)

    def run_array(self, index=0):
        self.index = index
        self.network_params, _ = augment_args(index, self.network_params)
        self.run()

    def enqueue(self):
        """
        1. save the current Experiment object to a pickle temp file
        2. create the batch file / hardware params
        3. sbatch the batch file
        """
        exp_file = f"experiments/experiment-{str(time()).replace('.', '')}.pkl"
        batch_text = f"""#!/bin/bash
#SBATCH --partition={self.hardware_params['partition']}
#SBATCH --cpus-per-task={self.hardware_params['n_cpu']}
#SBATCH --ntasks=1
#SBATCH --mem={self.hardware_params['memory']}
#SBATCH --output={self.hardware_params['stdout_path']}
#SBATCH --error={self.hardware_params['stderr_path']}
#SBATCH --time={self.hardware_params['time']}
#SBATCH --job-name={self.hardware_params['name']}
#SBATCH --mail-user={self.hardware_params['email']}
#SBATCH --mail-type=ALL
#SBATCH --chdir={self.hardware_params['dir']}
#SBATCH --nodelist={','.join(self.hardware_params['nodelist'])}
#SBATCH --array={self.hardware_params['array']}
. /home/fagg/tf_setup.sh
conda activate tf_bleeding5

python run.py --pkl {exp_file} --lscratch $LSCRATCH --id $SLURM_ARRAY_TASK_ID"""

        with open('experiment.sh', 'w') as fp:
            fp.write(batch_text)

        with open(exp_file, 'wb') as fp:
            pickle.dump(self, fp)

        self.batch_file = batch_text

        os.system(f'sbatch experiment.sh')


class Results:
    """
        includes:
            config
            experiment
            model data
    """

    def __init__(self, experiment, model_data):
        self.config = Config(experiment.hardware_params, experiment.network_params,
                             experiment.dataset_params, experiment.params)
        self.experiment = experiment
        self.model_data = model_data

    def summary(self):
        metrics = [key for key in self.model_data.history]
        patience = self.config.experiment_params['patience']
        epochs = len(self.model_data.history['loss'])
        performance_at_patience = {key: self.model_data.history[key][epochs - patience - 1]
                                   for key in metrics}

        index = 'n/a' if self.experiment.index is None else self.experiment.index
        run_params = dict_to_string(dict(self.experiment.run_args)) if isinstance(self.experiment.run_args,
                                                                                  dict) else None
        print(f"""
------------------------------------------------------------
Experimental Results Summary (Index: {index})
------------------------------------------------------------
Dataset Params: {dict_to_string(self.experiment.dataset_params)}

Network Params:  {dict_to_string(self.experiment.network_params)}
------------------------------------------------------------
Experiment Parameters: {dict_to_string(self.experiment.params)}

Experiment Runtime: {self.model_data.run_time}s

Epochs Run / Average Time: {epochs} / {self.model_data.run_time / len(self.model_data.history['loss'])}s

Performance At Patience: {dict_to_string(performance_at_patience)}
------------------------------------------------------------
Runtime Parameters: {run_params}
------------------------------------------------------------
        """)


def load_most_recent_results(d, n=1):
    """
    load the n most recent result files from the result directory
    :param d: directory from which to load the files
    :param n: number of recent results to load
    :return: a list of Results objects
    """
    results = []

    for file in sorted(os.listdir(d), key=lambda f: os.stat(os.path.join(d, f)).st_mtime, reverse=True):
        with open(os.path.join(d, file), 'rb') as fp:
            results.append(pickle.load(fp))
            n -= 1
            if not n:
                break

    return results


def load_most_recent_results_with_fnames(d, n=1):
    """
    load the n most recent result files from the result directory
    :param d: directory from which to load the files
    :param n: number of recent results to load
    :return: a list of Results objects
    """
    results = []
    fnames = []
    
    for file in sorted(os.listdir(d), key=lambda f: os.stat(os.path.join(d, f)).st_mtime, reverse=True):
        if os.path.isfile(os.path.join(d, file)):
            with open(os.path.join(d, file), 'rb') as fp:
                results.append(pickle.load(fp))
                fnames.append(fp)
                n -= 1
                if n == 0:
                    break

    return results, fnames