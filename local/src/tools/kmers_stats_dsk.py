#!/usr/env python 
import pandas as pd 
import seaborn as sns
import argparse 
import os, sys
from kneed import KneeLocator

def main():
    parser = argparse.ArgumentParser(description="Computes differential kmers (developed on DSK output)")
    parser.add_argument("-i", '--input', required=True, help="Kmers frequency table")
    parser.add_argument("-o", "--outprefix", required=True, help="Output prefix")
    parser.add_argument("-a1", "--assembly1", required=True, help="First assembly name")
    parser.add_argument("-a2", "--assembly2", required=True, help="Second assembly name")

    args = parser.parse_args()
    df = pd.read_csv(args.input, sep="\t", index_col=0)
    diff = df[args.assembly1] - df[args.assembly2]
    diff.to_csv(args.outprefix + ".diff", sep="\t")

    abs_diff = abs(diff)
    abs_diff.to_csv(args.outprefix + ".diff_abs", sep="\t")   
    print(abs_diff)
    x,y = sns.kdeplot(abs_diff).get_lines()[0].get_data()
    # find the knee
    kn = KneeLocator(x, y, curve='convex', direction='decreasing')
    knee_x = kn.knee
    #knee_y = kn.knee_y
    df_filter = df.loc[abs_diff[abs_diff > kn.knee].index]
    df_filter.to_csv(args.outprefix + ".freqs_filtered", sep="\t")

if __name__ == "__main__":
    sys.exit(main())