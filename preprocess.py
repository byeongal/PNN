import os
import sys
import getopt
import pickle

import multiprocessing as mp

import tqdm
import torch
import pandas as pd

import utils

def vectorize_payload(payload):
    payload = [ int(payload[i:i+2], 16) + 1 for i in range(0, len(payload), 2) ]
    payload.extend([0] * (1600 - len(payload)))
    return torch.tensor(payload, dtype=torch.long)

def event_dump(args):
    file_path, save_dir = args
    with open(file_path, 'rb') as f:
        event = pickle.load(f)
    _id = event['_id']
    payload = vectorize_payload(event['payload'])
    with open(os.path.join(save_dir, _id + '.dat'), 'wb') as f:
        pickle.dump(payload, f)
    return os.path.join(save_dir, _id + '.dat'), int(event['analyResult']) % 2

def main(argv):
    csv_path = None
    save_dir = None
    output_csv_path = None
    process = os.cpu_count()
    optlist, args = getopt.getopt(argv[1:], '', ['help', 'csv_path=', 'save_dir=', 'output_csv_path=', 'process='])
    for opt, arg in optlist:
        if opt == '--help':
            utils.train_help()
            sys.exit(0)
        elif opt == '--csv_path':
            csv_path = arg
        elif opt == '--save_dir':
            save_dir = arg
        elif opt == '--process':
            process = int(arg)
        elif opt == '--output_csv_path':
            output_csv_path = arg

    mp.freeze_support()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df = pd.read_csv(csv_path, header=None)

    fn_list = df[0].values
    target_dir_list = [save_dir] * len(fn_list)

    with open(output_csv_path, 'w') as f:
        with mp.Pool(process) as pool:
            for fn, label in tqdm.tqdm(pool.imap_unordered(event_dump, zip(fn_list, target_dir_list)), total=len(fn_list)):
                f.write(f'{fn},{label}\n')

if __name__ == '__main__':
    main(sys.argv)