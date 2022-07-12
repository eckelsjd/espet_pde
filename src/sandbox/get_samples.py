import pickle
from pathlib import Path
import itertools
import os


def load_all_pickles(filename):
    """Generator that loads all pickles"""
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def load_all_pickle_files(dirpath):
    """Return a generator over all items in all pickle files in a directory"""
    filenames = []
    with os.scandir(dirpath) as it:
        for entry in it:
            if entry.name.endswith('.pkl') and entry.is_file():
                filenames.append(load_all_pickles(entry.path))
    return itertools.chain(*filenames)


if __name__ == '__main__':
    sdir = Path('../../data/base/samples')
    sfile = sdir / 'samples.txt'
    fd = open(sfile, 'w')
    fd.write('d rc alpha h ra\n')
    ps = load_all_pickle_files(sdir)

    for p in ps:
        params = p[0]
        wstring = f"{params['Tip-to-Extractor Distance']} {params['Cone Radius of Curvature']} \
{params['Cone Half-Angle']} {params['Tip Height']} {params['Radius of Aperature']}\n"
        fd.write(wstring)

    fd.close()
