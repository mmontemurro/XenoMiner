#!/usr/bin/env python

import sys, os
import numpy as np
import argparse
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import h5py
from numpy import array

base_val = {"A": 0, "C":1, "G":2, "T":3}
def check_gaps(seq):
    for b in seq:
        if b not in base_val.keys():
            return True
    return False

def main():
    parser = argparse.ArgumentParser(description="Compute k-mers frequency matrix from fasta")
    parser.add_argument("-i", "--input_file", help="Input fasta file", required=True)
    parser.add_argument("-o", "--output_prefix", help="Output file", required=True)
    parser.add_argument("-g", "--discard_gaps",  action="store_true", default=False, help="If declared, discardes sequences which contain > 50 perc of Ns")
    parser.add_argument("-t", "--file_type",  help="File format (fasta/fastq). Default = fastq.", default="fasta")
    parser.add_argument("-m", "--min_l", help="Mimum sequence lentgh threshold. The shorter ones are discarded", default=50, type=int)
    parser.add_argument("-M", "--max_l", help="Maximun sequence lentgh threshold. The longer ones are discarded", default=150, type=int)

    args = parser.parse_args()
    
    min_l = args.min_l
    max_l = args.max_l

    seq_fasta = [s for s in SeqIO.parse(args.input_file, "fasta") if len(s) <= max_l and len(s) >= min_l]
    #hf = h5py.File(args.output_prefix + "_onehot.hd5", 'w')
    os.makedirs(args.output_prefix, exist_ok=True)

    i_sequence = 0
    for sequence in seq_fasta:
        seq = sequence.seq.upper()
        if args.discard_gaps == True:
            if check_gaps(seq):
                continue
        if (i_sequence % 1000 == 0):
            print("Reading %dth sequence." % (i_sequence + 1))
        
        data = [b for b in sequence ]
        values = array(data)

        # convert A,C,G,T to 0,1,2,3
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)

        # transform sequence to one-hot matrix
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        #store to hdf5 file
        seq_id = os.path.basename(sequence.id)
        #hf.create_dataset(seq_id, data=onehot_encoded)
        np.savetxt(os.path.join(args.output_prefix, seq_id + ".csv"), onehot_encoded, delimiter=",")
        
        i_sequence = i_sequence + 1

    #hf.close()

if __name__ == "__main__":
    sys.exit(main())

