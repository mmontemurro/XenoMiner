#!/usr/env python

import os, sys
import argparse
import pickle
from scipy.stats import pearsonr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Computes Pearson correlation index between the distributions of kmers on chromosomes of the same species")
    parser.add_argument("-i", "--input", required=True, help="Kmer count files", nargs="+")
    parser.add_argument("-c", "--chromosomes", required=True, help="Chromosome list", nargs="+")
    parser.add_argument("-o", "--outprefix", required=True, help="Output path")

    args = parser.parse_args()

    #input_files = []
    #for c in args.chromosomes:
    #    f = args.prefix + "_" + c + "_" + args.kvalue + "mer_count"
    #    input_files.append(f)
    
    counts = []
    for f in args.input:
        with open(f, "rb") as filehandle:
            kmer_counts = pickle.load(filehandle)
        count = kmer_counts.A[0, :].sum()
        counts.append(kmer_counts.A[0, :]/count)

    
    df = pd.DataFrame(np.corrcoef(counts), columns=args.chromosomes, index=args.chromosomes)

    outfile = args.outprefix + "_pairwise_chrom_pearson.csv"
    df.to_csv(outfile)

    df = df.round(2)

    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(df, annot=True, annot_kws={"size":8}, ax=ax)
    outfig = args.outprefix + "_pairwise_chrom_pearson_heatmap.png"
    plt.savefig(outfig)
    plt.close('all')

if __name__ == "__main__":
    sys.exit(main())
