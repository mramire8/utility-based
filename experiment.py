import argparse
import utilities.configutils as cfgutils
from experiment.base import Experiment

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument('--train',
                metavar='TRAIN',
                default="20news",
                choices=['imdb', 'amt_imdb', 'sraa', '20news', 'twitter','arxiv', 'twitter-gender'],
                help='training data (libSVM format)')

ap.add_argument('--verbose',
                action='store_true',
                help='to print progress of experiment')

ap.add_argument('--debug',
                action='store_true',
                help='to print query details of experiment')


ap.add_argument('--config',
                metavar='CONFIG_FILE',
                type=str,
                default='./default2.cfg',
                help='Experiment configuration file')


def main():
    from time import time
    t0 = time()
    args = ap.parse_args()
    print args.train
    config = cfgutils.get_config(args.config)
    experiment = Experiment(args.train, config, verbose=args.verbose, debug=args.debug)
    experiment.start()
    t1 = time()
    print "\nElapsed time: %.3f secs (%.3f mins)" % ((t1-t0), (t1-t0)/60)

if __name__ == "__main__":
    main()