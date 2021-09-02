#!/usr/env python

import os, sys
import argparse
import pickle 
from scipy.sparse import csr_matrix

def main():
    parser = argparse.ArgumentParser(description="Filters kmers count array, keeping only those which are found to be mostly differential among species.")
    parser.add_argument("-i", "--input", help="Chromosome count file", required=True)
    parser.add_argument("-d", "--differential", help="File containing the differences computed among human kmer frequencies and mouse ones", required=True)
    parser.add_argument("-t", "--threshold", help="Threshold above which kmers are kept", required=True, type=float)
    parser.add_argument("-o", "--outfile", help="Output file", required=True)

    args = parser.parse_args()

    # Read array containing differential frequencies
    with open(args.differential, "rb") as f:
        diff = pickle.load(f)

    # Read kmers count file
    with open(args.input, "rb") as f:
        count = pickle.load(f)

    kmers_count_array = count.A[0, :]
    kmers_count_filter = kmers_count_array[diff > args.threshold]

    with open(args.outfile, "wb") as f:
        kmers_count_sparse = csr_matrix(kmers_count_filter)
        pickle.dump(kmers_count_sparse, f)


if __name__ == "__main__":
    sys.exit(main())

    