import torch
import torch.nn as nn
from collections import OrderedDict
from .utils import tensor_to_state, state_to_tensor


class Model(nn.Module):
    """
    Model base class
    """
    def __init__(self):
        super(Model, self).__init__()
        self.drop_last = False

    def updated_state_dict(self):
        """
        Parameters that are uploaded to the server for aggregating
        By default, it is all the parameters
        :return:
        """
        return self.state_dict()

    def get_params_tensor(self):
        state = self.uploaded_state_dict()
        tensor = state_to_tensor(state)
        return tensor

    def load_params_tensor(self, tensor):
        template = self.uploaded_state_dict()
        new_state = tensor_to_state(tensor, template)
        self.load_state_dict(new_state, strict=False)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_parameter_tensor(self):
        trainable_parameters = [tensor.view(-1) for tensor in self.parameters() if tensor.requires_grad]
        return torch.cat(trainable_parameters)


def test():
    model = Model()
    print(model.state_dict())  # OrderedDict()
    print(model.uploaded_state_dict())  # OrderedDict()
    # print(model.personal_state_dict())  # OrderedDict()
    # print(model.freezed_state_dict())  # OrderedDict()
