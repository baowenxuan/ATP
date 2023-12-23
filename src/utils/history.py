from .load_and_save import pickle_save


class History(object):
    """
    Saving the training history for later use.
    Example:
    history.data = {
        'args': args
        'train_loss': [1.0, 0.9, 0.8, ...]
        'test_loss': [1.1, 1.0, 0.9, ...]
        'train_acc': [0.40, 0.43, 0.46, ...]
        'test_acc': [0.37, 0.40, 0.43, ...]
    }
    """

    def __init__(self):
        self.data = {}

    def append(self, record):
        """
        Add a line of record (usually one record for each time step).
        :param record: a dictionary
        :return:
        """
        for key, value in record.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

    def concat(self, attributes):
        """
        Add some attributes
        :param kwargs:
        :return:
        """
        for key, value in attributes.items():
            assert key not in self.data  # the added attributes should not overwrite any current attributes.
            self.data[key] = value

    def save(self, path, mode='wb'):
        pickle_save(obj=self.data, file=path, mode=mode)
