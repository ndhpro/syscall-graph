from multiprocessing import cpu_count
from glob import glob

SEED = 42
N_JOBS = cpu_count()
RP_PATHS = glob(
    '/media/vserver12/data/vsandbox_report/final_report_*/*/')
ELF_PATHS = glob('/home/vserver12/Downloads/botnet/*/*')
