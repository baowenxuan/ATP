import os
import pickle


def pickle_load(file, multiple=False):
    if not multiple:
        with open(file, 'rb') as f:
            return pickle.load(f)
    else:  # recursive run
        objects = []
        with open(file, 'rb') as f:
            while True:
                try:
                    objects.append(pickle.load(f))
                except EOFError:
                    break
            return objects


def pickle_save(obj, file, mode='ab'):
    if not os.path.exists(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))
    with open(file, mode) as f:
        pickle.dump(obj, f)
