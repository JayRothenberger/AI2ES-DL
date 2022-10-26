import pickle
from util import Experiment

# TODO: add parser for lscratch and experiment id

if __name__ == "__main__":
    with open('experiment') as fp:
        exp = pickle.load(fp)

    exp.run()
