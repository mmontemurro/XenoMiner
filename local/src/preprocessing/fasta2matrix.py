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

def normalize_vector (normalize_method,
                      k,
                      vector,
                      kmer_list):
    # Do nothing if there's no normalization.
    if (normalize_method == "none"):
        return(vector)
    # Initialize all vector lengths to zeroes.
    vector_lengths = 0
    # Compute sum or sum-of-squares separately for each k.
    num_kmers = len(kmer_list)
    for i_kmer in range(0, num_kmers):
        count = vector[i_kmer]
        if (normalize_method == "frequency"):
            vector_lengths += count
        elif (normalize_method == "unitsphere"):
            vector_lengths += count * count
    # Square root each sum if we're doing 2-norm.
    if (normalize_method == "unitsphere"):
        vector_lengths = math.sqrt(vector_lengths[k])
    # Divide through by each sum.
    return_value = []
    for i_kmer in range(0, num_kmers):
        count = vector[i_kmer]
        if (vector_lengths == 0):
            return_value.append(0)
        else:
            return_value.append(float(count) / float(vector_lengths))
    return(return_value)

##############################################################################
# Make a copy of a given string, substituting one letter.
def substitute (position,
                letter,
                string):

    return_value = ""
    if (position > 0):
        return_value = return_value + string[0:position]
    return_value = return_value + letter
    if (position < (len(string) - 1)):
        return_value = return_value + string[position+1:]
                   
    return(return_value)


##############################################################################
def make_sequence_vector (sequence,
                          normalize_method,
                          k,
                          kmer_list,
                          pseudocount):
    # Make an empty counts vector.
    kmer_counts = {}
    #for i_bin in range(0, num_bins):
    #    kmer_counts.append({})
    # Iterate along the sequence.
    seq_length = len(sequence) - k + 1
    for i_seq in range(0, seq_length):
        # Compute which bin number this goes in.
        #bin_num = compute_bin_num(num_bins, i_seq, k, numbers)
        # Extract this k-mer.
        kmer = sequence[i_seq : i_seq + k]
        # If we're doing reverse complement, store the count in the
        # the version that starts with A or C.
        #if (revcomp == 1):
        #    rev_kmer = find_revcomp(kmer, revcomp_dictionary)
        #    if (cmp(kmer, rev_kmer) > 0):
        #        kmer = rev_kmer
        # Increment the count.
        if (kmer in kmer_counts.keys()):
            kmer_counts[kmer] += 1
        else:
            kmer_counts[kmer] = 1
    # Build the sequence vector.
    sequence_vector = []
    #for i_bin in range(0, num_bins):
    #    for kmer in kmer_list:
    #        if (kmer_counts[i_bin].has_key(kmer)):
    #            sequence_vector.append(kmer_counts[i_bin][kmer] + pseudocount)
    #        else:
    #            sequence_vector.append(pseudocount)
    for kmer in kmer_list:
        if (kmer in kmer_counts.keys()):
            sequence_vector.append(kmer_counts[kmer] + pseudocount)
        else:
            sequence_vector.append(pseudocount)
    # Normalize it
    return_value = normalize_vector(normalize_method,
                                    k,
                                    sequence_vector,
                                    kmer_list)
    return(return_value)

#Reads 1 sequence from a fasta file
def read_fasta_sequence(fasta_file):
    # Read 1 byte.  
    first_char = fasta_file.read(1)
    # If it's empty, we're done.
    if (first_char == ""):
        return([""])
    # If it's ">", then this is the first sequence in the file.
    elif (first_char == ">"):
        line = ""
    else:
        line = first_char
    # Read the rest of the header line.
    seq_id = line + fasta_file.readline()
    seq_id = seq_id.rstrip('\n')
    # Get the rest of the ID.
    #words = line.split()
    #if (len(words) == 0):
    #  sys.stderr.write("No words in header line (%s)\n" % line)
    #  sys.exit(1)
##    id = words[0]
    #id = words[1].split("=")[1]        
    # Read the sequence, through the next ">".
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
    for letter in sequence:
        if (letter != " "):
            clean_sequence = clean_sequence + letter
    sequence = clean_sequence.upper()       
    return [seq_id, sequence]

def main():
    # Define the command line usage.
"""
    Options:
    -upto       Use all values from k up to the specified k.
    -revcomp    Collapse reverse complement counts.
    -normalize [frequency|unitsphere] Normalize counts to be 
                frequencies or project onto unit sphere.  With -upto,
                normalization is done separately for each k.
    -alphabet <string> Set the alphabet arbitrarily.
    -mismatch <value>  Assign count of <value> to k-mers that 
                       are 1 mismatch away.
   -pseudocount <value>  Assign the given pseudocount to each bin.
"""

parser = argparse.ArgumentParser(description="Compute k-mers frequency matrix from fasta")
parser.add_argument("-k", "--k_value", help="k value", required=True)
parser.add_argument("-i", "--input_file", help="Input fasta file", required=True)
parser.add_argument("-o", "--output_file", help="Output file", required=True)
parser.add_argument("-d", "--kmers_dict", help="k-mers dictionary", required=True)
parser.add_argument("-a", "--alphabet", help="Set the alphabet arbitrarily", default="ACGT")
parser.add_argument("-n", "--normalize", help="Normalize counts to be frequencies or project onto unit sphere.", choices=["frequency", "unitsphere"], default=None)
parser.add_argument("-p", "--pseudocount", help="Assign the given pseudocount to each matrix cell", default=0.1)
parser.add_argument("-m", "--mismatch", help="Assign count of <value> to k-mers that are 1 mismatch away", default=0)

args = parser.parse_args()

#upto = 0
#revcomp = 0
k = args.k_value
normalize_method = args.normalize
alphabet = args.alphabet
mismatch = args.mismatch
pseudocount = args.pseudocount

# Check for reverse complementing non-DNA alphabets.
#if ((revcomp == 1) and (alphabet != "ACGT")):
#  sys.stderr.write("Attempted to reverse complement ")
#  sys.stderr.write("a non-DNA alphabet (%s)\n" % alphabet)

# Make a list of all values of k.
#k_values = []
#if (upto == 1):
#  start_i_k = 1
#else:
#  start_i_k = k
#k_values = range(start_i_k, k+1)

  
# Make a list of all k-mers.
with open(args.kmers_dict, 'rb') as filehandle:
    kmer_list = pickle.load(filehandle)

# Set up a dictionary to cache reverse complements.
#revcomp_dictionary = {}

# Use lexicographically first version of {kmer, revcomp(kmer)}.
#if (revcomp == 1):
#  new_kmer_list = []
#  for kmer in kmer_list:
#      rev_kmer = find_revcomp(kmer, revcomp_dictionary)
#      if (cmp(kmer, rev_kmer) <= 0):
#          new_kmer_list.append(kmer)
#  kmer_list = new_kmer_list;
#  sys.stdout.write("Reduced to %d kmers.\n" % len(kmer_list))

# Print the corner of the matrix.

outfile=open(args.output_file,"wb")

# Print the title row.

outfile.write("seq_id,")


fasta_file = open(args.input_file, "r")
#for i_bin in range(1, num_bins+1):
for kmer in kmer_list:
  #if (num_bins > 1):
    #outfile.write("%s-%d," % (kmer, i_bin))
      #i+=1
  #else:
    if(kmer==kmer_list[len(kmer_list)-1]):
      outfile.write("%s" % kmer)
    else:
      outfile.write("%s," %kmer)
      #i+=1

outfile.write("\n")
# Read the first sequence.
[id, sequence] = read_fasta_sequence(fasta_file)


# Iterate till we've read the whole file.
i_sequence = 1
vett=np.zeros(len(kmer_list),dtype=int)
while (id != ""):

  # Tell the user what's happening.
  if (i_sequence % 1000 == 0):
    print("Reading %dth sequence." % i_sequence)

  # Compute the sequence vector.
  vector = make_sequence_vector(sequence,
                                normalize_method,
                                k,
                                kmer_list,
                                pseudocount)

  # Print the formatted vector.
  outfile.write("%s," % id)
  
  count=len(kmer_list)

  for element in vector:
    if(count!=1):
        outfile.write("%d," % element)
    else:
        outfile.write("%d" % element)
    count = count-1
        
  outfile.write("\n")
  # Read the next sequence.
  [id, sequence] = read_fasta_sequence(fasta_file)
  i_sequence += 1
# Close the file.
  i=0


outfile.close()
fasta_file.close()