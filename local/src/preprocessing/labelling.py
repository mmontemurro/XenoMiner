import sys
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Concatenates human and murine matrices, adding the proper label")
    parser.add_argument("-h", "--human", help="Human matrix", required=True)
    parser.add_argument("-m", "--murine", help="Murine matrix", required=True)
    parser.add_argument("-o", "--output_file", help="Output file", required=True)

    args = parser.parse_args()

    df_h = pd.read_csv(args.human, index_col=0)
    df_m = pd.read_csv(args.murine, index_col=0)

    df = pd.DataFrame(df_h, df_m)

    df.to_csv(args.output_file)

if __name__ == "__main__":
    sys.exit(main())