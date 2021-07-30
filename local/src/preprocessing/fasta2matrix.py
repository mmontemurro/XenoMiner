#!/usr/bin/env python

#This file contains the code to preprocess a fasta file containing reads, to produce the input matrix for the model
# Encoding: word embedding
# Vocabulary: all possibile k-mers of k bases
# Embedding criteria: skip-gram with negative sampling model (https://www.biorxiv.org/content/10.1101/726729v1, https://www.liebertpub.com/doi/10.1089/cmb.2018.0174)
# LocalitySensity Hashing (LSH)

import sys
import math
import numpy as np
import argparse
import pickle
from Bio import SeqIO
from scipy.sparse import csr_matrix, diags

def list2dict(a):
    res_dct = {a[i] : i for i in range(0, len(a) ) }
    return res_dct

def normalize_matrix (matrix):
    #create a sparse diagonal matrix from the reciprocals of row sums
    d = diags(1/matrix.sum(axis=1).A.ravel())

    #multiply the diagonal matrix with the matrix
    n = (d @ matrix)
    return n


##############################################################################
def make_sequence_vector (sequence,
                          k,
                          kmer_dict,
                          indptr,
                          indices,
                          data):

    # Iterate along the sequence, find kmers and build the sparce count vector.
    seq_length = len(sequence) - k + 1
    for i_seq in range(0, seq_length):
        # Extract this k-mer.
        kmer = sequence[i_seq : i_seq + k]
        if kmer in kmer_dict.keys():
            #There may be some kmers containing N's, so we skip them
            index = kmer_dict[kmer]
            indices.append(index)
            data.append(1)

    indptr.append(len(indices))
    return [indptr, indices, data]

#Reads 1 sequence from a fasta file
def read_fasta_sequence(fasta_file):
    # Read 1 byte.  
    first_char = fasta_file.read(1)
    # If it's empty, we're done.
    if (first_char == ""):
        return("")
    # If it's ">", then this is the first sequence in the file.
    elif (first_char == ">"):
        line = ""
    else:
        # the previous iteration has already consumed the ">"
        line = first_char
    # Read the rest of the header line.
    seq_id = line + fasta_file.readline()
    seq_id = seq_id.rstrip('\n')
    # Read the sequence, through the next ">" or the end of file.
    first_char = fasta_file.read(1)
    sequence = ""
    while ((first_char != ">") and (first_char != "")):
        if (first_char != "\n"): # Handle blank lines.
            line = fasta_file.readline()
            sequence = sequence + first_char + line
        first_char = fasta_file.read(1)
    # Remove EOLs.
    clean_sequence = ""
    for letter in sequence:
        if (letter != "\n"):
            clean_sequence = clean_sequence + letter
    sequence = clean_sequence
    # Remove spaces.
    clean_sequence = ""
    N_counts = 0
    for letter in sequence:
        if (letter != " "):
            clean_sequence = clean_sequence + letter
            if letter == 'N' or letter == 'n':
                N_counts += 1
    seq_length = len(sequence)
    if N_counts <= (0.5*seq_length):
        sequence = clean_sequence.upper()     
    else:
        sequence = None  
    return sequence

def check_gaps(seq):
    count = 0
    for b in seq:
        if b == 'N':
            count += 1
    return count

def main():
    parser = argparse.ArgumentParser(description="Compute k-mers frequency matrix from fasta")
    parser.add_argument("-k", "--k_value", help="k value", required=True, type=int)
    parser.add_argument("-i", "--input_file", help="Input fasta file", required=True)
    parser.add_argument("-o", "--output_file", help="Output file", required=True)
    parser.add_argument("-d", "--kmers_dict", help="k-mers dictionary", required=True)
    parser.add_argument("-a", "--alphabet", help="Set the alphabet arbitrarily", default="ACGT")
    parser.add_argument("-n", "--discard_gaps", action="store_true", default=False, help='If declared, discards reads made for more than 50%% of N')
    parser.add_argument("-f", "--kmers_frequency", action="store_true", default=False, help="If declared, normalizes kmer counts by computing threir frequency in each read")

    args = parser.parse_args()

    k = args.k_value
    alphabet = args.alphabet
  
    # Make a list of all k-mers.
    with open(args.kmers_dict, 'rb') as filehandle:
        kmer_list = pickle.load(filehandle)
    kmer_dict = list2dict(kmer_list)


    fasta_file = SeqIO.parse(open(args.input_file),'fasta')
    # Iterate till we've read the whole file and generate a sparse count matrix
    indptr = [0]
    indices = []       
    data = []

    i_sequence = 0
    for sequence in fasta_file:
        # Tell the user what's happening.
        if args.discard_gaps == True:
            Ncount = check_gaps(sequence.seq)
            if Ncount > (0.5*len(sequence.seq)):
                continue
        if (i_sequence % 1000 == 0):
            print("Reading %dth sequence." % (i_sequence + 1))
        # Compute the sparse count vector.
        [indptr, indices, data] = make_sequence_vector(sequence.seq,
                                k,
                                kmer_dict,
                                indptr,
                                indices,
                                data)    # Close the file.
        i_sequence += 1

    #count_matrix = csr_matrix((data, indices, indptr), shape=(len(list(fasta_file)), len(kmer_list)), dtype=int)
    count_matrix = csr_matrix((data, indices, indptr), dtype=int)
    if args.kmers_frequency == True:
        frequency_matrix = normalize_matrix(count_matrix)
        with open(args.output_file, "wb") as f:
            pickle.dump(frequency_matrix, f)
    else:
        with open(args.output_file, "wb") as f:
            pickle.dump(count_matrix, f)
    

if __name__ == "__main__":
    sys.exit(main())