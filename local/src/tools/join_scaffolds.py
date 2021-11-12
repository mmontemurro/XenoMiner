import sys
import argparse
import pickle
from scipy.sparse import vstack

def main():
    parser = argparse.ArgumentParser(description="Generates label array per chromosome")
    parser.add_argument("-R1", "--read_1", help="R1 matrix", required=True)
    parser.add_argument("-R2", "--read_2", help="R2 matrix", required=True)
    parser.add_argument("-o", "--outpath", help="Output path", required=True)

    args = parser.parse_args()

    with open(args.read_1, "rb") as f:
        r1 = pickle.load(f)

    with open(args.read_2, "rb") as f:
        r2 = pickle.load(f)

    dataset = vstack((r1, r2))

    with open(args.outpath, "wb") as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    sys.exit(main())