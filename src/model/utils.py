import torch
from copy import deepcopy


def state_to_tensor(model_state_dict):
    """
    Convert a state dict to a concatenated tensor
    Note: it is deep copy, since torch.cat is deep copy
    :param model_state_dict:
    :return:
    """
    tensor = [t.view(-1) for t in model_state_dict.values()]
    tensor = torch.cat(tensor)
    return tensor


def tensor_to_state(tensor, model_state_dict_template):
    """
    Convert a tensor back to state dict.
    Note: apply deepcopy inside the function. Only use the input state dict as a template
    :param model_state_dict:
    :return:
    """
    curr_idx = 0
    model_state_dict = deepcopy(model_state_dict_template)
    for key, value in model_state_dict.items():
        numel = value.numel()
        shape = value.shape
        model_state_dict[key].copy_(tensor[curr_idx:curr_idx + numel].view(shape))
        curr_idx += numel

    return model_state_dict


def model_numel(model, typ='all'):
    """
    Calculate the number of parameters of a
    :param model:
    :return:
    """
    num = 0
    if typ == 'all':
        for tensor in model.state_dict().values():
            num += tensor.numel()
    elif typ == 'uploaded':
        for tensor in model.uploaded_state_dict().values():
            num += tensor.numel()
    # elif typ == 'personal':
    #     for tensor in model.personal_state_dict().values():
    #         num += tensor.numel()
    # elif typ == 'freezed':
    #     for tensor in model.freezed_state_dict().values():
    #         num += tensor.numel()
    else:
        raise NotImplementedError('Unknown type of parameter to count. ')

    return num



def test():
    from torch.nn import Linear
    # from copy import deepcopy
    # model1 = Linear(5, 3)
    # print(model1.weight)
    # model2 = Linear(5, 3)
    # print(model2.weight)
    #
    # state = model1.state_dict()
    # model2.load_state_dict(state)
    # # print(model1.weight)
    # with torch.no_grad():
    #     model1.weight.copy_(torch.ones(3, 5))
    # print(model1.weight)
    # print(state)
    # print(model2.weight)

    model = Linear(5, 3)
    model2 = Linear(5, 3)
    state = model.state_dict()
    model.load_state_dict(state)
    state['weight'][0, 0] = 10
    # print(state)
    print(model.weight[0, 0])

    # model.to('cuda')
    # print('Model:', model.weight)
    #
    # state = model.state_dict()
    # print('State:', state)
    #
    # tensor = state_to_tensor(state)
    # print('Tensor:', tensor)
    #
    # # Model --shallow--> State ----deep---> Tensor
    # state['weight'][0, 0] = 10
    # print('Model:', model.weight)
    # print('State:', state)
    # print('Tensor:', tensor)
    #
    # # change state, the model changes, but the tensor does not.
    # state = tensor_to_state(tensor, state)
    # model.load_state_dict(state)
    #
    # # Tensor ----deep---> State --shallow--> Tensor
    # state['weight'][0, 1] = 20
    # print('Model:', model.weight)
    # print('State:', state)
    # print('Tensor:', tensor)
    #
    # state['weight'][0, 2] = 30
    # print('Model:', model.weight)






    # state = deepcopy(model.state_dict())
    # tensor = state_to_tensor(state)
    # print(tensor.shape, tensor.device)
    # print(tensor)
    # tensor[0] = 9
    # print(tensor)
    #
    # state = tensor_to_state(tensor, state)
    # print(state)
    # tensor[0] = 4
    # print(tensor)
    # print(state)
    #
    # print('Number of parameters:', model_numel(model, 'all'))
