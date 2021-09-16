import sys
import argparse
import pickle
from scipy.sparse import vstack

def main():
    parser = argparse.ArgumentParser(description="Generates label array per chromosome")
    parser.add_argument("-hu", "--human", help="Human matrix", required=True)
    parser.add_argument("-m", "--mouse", help="Murine matrix", required=True)
    parser.add_argument("-o", "--outprefix", help="Output prefix", required=True)

    args = parser.parse_args()

    # read both matrix size
    # generate an array of labels
    # concatenate the two matrices
    with open(args.human, "rb") as f:
        h = pickle.load(f)

    with open(args.mouse, "rb") as f:
        m = pickle.load(f)

    labels = []
    for i in range(h.shape[0]):
        labels.append("h")
    for i in range(m.shape[0]):
        labels.append("m")

    dataset = vstack((h, m))

    with open(args.outprefix + ".dat", "wb") as f:
        pickle.dump(dataset, f)

    with open(args.outprefix + ".lab", "wb") as f:
        pickle.dump(labels, f)
    
if __name__ == "__main__":
    sys.exit(main())