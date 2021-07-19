#!/usr/bin/env python

# This file contains methods to generate the kmers dictionary
import sys, os
import argparse
import pickle

def make_kmer_list (k, alphabet):

    # Base case.
    if (k == 1):
        return(alphabet)

    # Handle k=0 from user.
    if (k == 0):
        return([])

    # Error case.
    if (k < 1):
        sys.stderr.write("Invalid k=%d" % k)
        sys.exit(1)

    # Precompute alphabet length for speed.
    alphabet_length = len(alphabet)

    # Recursive call.
    return_value = []
    for kmer in make_kmer_list(k-1, alphabet):
        for i_letter in range(0, alphabet_length):
            return_value.append(kmer + alphabet[i_letter])
              
    return return_value

def main():
    parser = argparse.ArgumentParser(description="Generates dictionary of kmers for a given k")
    parser.add_argument("-k", "--kmer_length", required=True, help="Desired kmer lenght", type=int)
    parser.add_argument("-a", "--alphabet", required=True, help="Bases alphabet", nargs="+")
    parser.add_argument("-o", "--out_path", required=True, help="Ouput directory path")

    args = parser.parse_args()

    dictionary = make_kmer_list(args.kmer_length, args.alphabet)

    outfile = os.path.join(args.out_path, str(args.kmer_length) + "mer_dictionary")
    #store the kmers dictionary in binary format (faster)
    with open(outfile, "wb") as f:
        pickle.dump(dictionary, f)


    #to read the dictionary
    #with open('listfile.data', 'rb') as filehandle:
    #   placesList = pickle.load(filehandle)

        
if __name__ == "__main__":
    sys.exit(main())
