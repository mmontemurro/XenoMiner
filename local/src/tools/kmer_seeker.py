import sys
import argparse
import numpy as np
import pickle
from Bio import SeqIO
from scipy.sparse import csr_matrix




def main():
    parser = argparse.ArgumentParser(description="Compute k-mers frequency matrix from fasta")
    parser.add_argument("-k", "--k_value", help="k value", required=True, type=int)
    parser.add_argument("-i", "--input_file", help="Input fasta file", required=True)
    parser.add_argument("-o", "--output_file", help="Output file", required=True)
    parser.add_argument("-d", "--kmers_dict", help="k-mers dictionary", required=True)
    parser.add_argument("-a", "--alphabet", help="Set the alphabet arbitrarily", default="ACGT")
    parser.add_argument("-m", "--mismatch", help="Assign count of <value> to k-mers that are 1 mismatch away", default=0)

    args = parser.parse_args()

    k = args.k_value
    alphabet = args.alphabet
    mismatch = args.mismatch
  
    # Make a list of all k-mers.
    with open(args.kmers_dict, 'rb') as filehandle:
        kmer_list = pickle.load(filehandle)
    kmer_dict = list2dict(kmer_list)


    fasta_file = SeqIO.parse(open(input_file),'fasta')


    # Iterate till we've read the whole file and generate a sparse count matrix
    indptr = [0]
    indices = []       
    data = []

    for sequence in fasta_file:
        # Tell the user what's happening.
        if (i_sequence % 1000 == 0):
            print("Reading %dth sequence." % i_sequence)
        # Compute the sparse count vector.
        [indptr, indices, data] = make_sequence_vector(sequence.seq,
                                k,
                                kmer_dict,
                                indptr,
                                indices,
                                data)    # Close the file.

    count_matrix = csr_matrix((data, indices, indptr), shape=(len(list(fasta_file)), len(kmer_list)), dtype=int)
    #frequency_matrix = normalize_matrix(count_matrix)
    
    with open(args.output_file, "wb") as f:
        pickle.dump(frequency_matrix, f)
    

if __name__ == "__main__":
    sys.exit(main())
