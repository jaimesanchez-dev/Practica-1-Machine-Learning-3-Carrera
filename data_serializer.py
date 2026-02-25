import pickle as pkl

class DataSerializer:
    @staticmethod
    def deserialize(file):
        with open(file, 'rb') as fd:
            data = pkl.load(fd)

        return data