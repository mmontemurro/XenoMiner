#!/usr/env python

import os, sys
import argparse
import pickle
from scipy.stats import pearsonr
from scipy.sparse import csr_matrix

def main():
    parser = argparse.ArgumentParser(description="Sums kmer counts per species and computes Pearson correlation index")
    parser.add_argument("-H", "--human", required=True, help="Human kmer count files", nargs="+")
    parser.add_argument("-M", "--mouse", required=True, help="Mouse kmer count files", nargs="+")
    parser.add_argument("-O", "--outdir", required=True, help="Output path")
    parser.add_argument("-K", "--kvalue", required=True, help="K value")
    
    args = parser.parse_args()


    #sum human chromosome kmer counts
    kmer_counts_h = []
    for h in args.human:
        with open(h, "rb") as f:
            m = pickle.load(f)
        if len(kmer_counts_h) == 0:
            kmer_counts_h = m.A[0, :]
        else:
            kmer_counts_h = kmer_counts_h + m.A[0,:]
    count = kmer_counts_h.sum()
    kmer_counts_h = kmer_counts_h/count

    kmer_counts_h_sparse = csr_matrix(kmer_counts_h)
    with open( os.path.join(args.outdir, "GRCh38_{}mer_count".format(args.kvalue)), "wb") as f:
        pickle.dump(kmer_counts_h_sparse, f)

    #sum mouse chromosome kmer counts
    kmer_counts_m = []
    for h in args.mouse:
        with open(h, "rb") as f:
            m = pickle.load(f)
        if len(kmer_counts_m) == 0:
            kmer_counts_m = m.A[0, :]
        else:
            kmer_counts_m = kmer_counts_m + m.A[0,:]
    
    count = kmer_counts_m.sum()
    kmer_counts_m = kmer_counts_m/count

    kmer_counts_m_sparse = csr_matrix(kmer_counts_m)
    with open( os.path.join(args.outdir, "GRCm38_{}mer_count".format(args.kvalue)), "wb") as f:
        pickle.dump(kmer_counts_m_sparse, f)

    s, p = pearsonr(kmer_counts_h, kmer_counts_m)

    with open( os.path.join(args.outdir, "GRCh38_GRCm38_{}mer_pearson_corr".format(args.kvalue)), "w" ) as f:
        f.write("pearson_corr_coeff\tpvalue\n{}\t{}\n".format(s, p))


if __name__ == "__main__":
    sys.exit(main())