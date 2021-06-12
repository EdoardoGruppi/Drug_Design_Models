import sys
import argparse
from hgraph import *
from rdkit import Chem
from tqdm.contrib.concurrent import process_map


def process(data):
    vocabulary = set()
    for line in data:
        s = line.strip("\r\n ")
        hmol = MolGraph(s)
        for node, attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocabulary.add(attr['label'])
            for i, s in attr['inter_label']:
                vocabulary.add((smiles, s))
    return vocabulary


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ncpu', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    data = [mol for line in sys.stdin for mol in line.split()[:2]]
    data = list(set(data))

    # batch_size = len(data) // args.ncpu + 1
    batch_size = args.batch_size
    batches = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]

    vocab_list = process_map(process, batches)
    vocabulary = [(x, y) for vocabulary in vocab_list for x, y in vocabulary]
    vocabulary = list(set(vocabulary))

    for x, y in sorted(vocabulary):
        print(x, y)
