import json
import subprocess

from joblib import Parallel, delayed
from glob import glob
from tqdm import tqdm
from constants import *

cmd = 'python FEATHER/src/main.py --model-type FEATHER-G --graphs data/scg-for-graph2vec/ --output data/feather.csv'
p = subprocess.Popen(cmd, shell=True)
p.wait()
