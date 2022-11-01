import pickle
from supervised.util import Experiment
import argparse


def create_parser():
    """
    Create argument parser
    """
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='CNN', fromfile_prefix_chars='@')

    parser.add_argument('--lscratch', type=str, default=None, help="Local scratch partition")
    parser.add_argument('--id', type=int, default=0, help="slurm array task id")
    parser.add_argument('--pkl', type=str, default="", help="pickle file name")

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.pkl, 'rb') as fp:
        exp = pickle.load(fp)

    exp.run_args = args

    exp.run_array(args.id)
