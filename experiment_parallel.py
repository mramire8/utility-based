import argparse
import utilities.configutils as cfgutils
from experiment.experiment_parallel import ExperimentJobs

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawTextHelpFormatter)

ap.add_argument('--verbose',
                action='store_true',
                help='to print progress of experiment')

ap.add_argument('--debug',
                action='store_true',
                help='to print query details of experiment')

ap.add_argument('--profile',
                action='store_true',
                help='to profile code execution')

ap.add_argument('--config',
                metavar='CONFIG_FILE',
                type=str,
                default='./default.cfg',
                help='Experiment configuration file')
ap.add_argument('--njobs',
                type=int,
                default=1,
                help='to profile code execution')

def main():
    from time import time
    t0 = time()
    args = ap.parse_args()

    config = cfgutils.get_config(args.config)
    experiment = ExperimentJobs(config, verbose=args.verbose, debug=args.debug)
    experiment.start(n_jobs=args.njobs)
    t1 = time()
    print "\nElapsed time: %.3f secs (%.3f mins)" % ((t1-t0), (t1-t0)/60)

if __name__ == "__main__":
    main()