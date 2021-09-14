import sys
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Generates label array per chromosome")
    parser.add_argument("-h", "--human", help="Human matrix", required=True)
    parser.add_argument("-m", "--mouse", help="Murine matrix", required=True)
    parser.add_argument("-o", "--output_file", help="Output file", required=True)

    args = parser.parse_args()

    # read both matrix size
    # generate an array of labels
    # concatenate the two matrices
    df_h = pd.read_csv(args.human, index_col=0)
    df_h['label'] = "h"
    df_m = pd.read_csv(args.murine, index_col=0)
    df_m['label'] = 'm'

    df = pd.DataFrame(df_h, df_m)

    df.to_csv(args.output_file)

if __name__ == "__main__":
    sys.exit(main())