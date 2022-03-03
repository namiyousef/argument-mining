import torch

def _move(data, device='cpu'):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: tensor.to(device) for key, tensor in data.items()}
    elif isinstance(data, list):
        raise NotImplementedError('Currently no support for tensors stored in lists.')
    else:
        raise TypeError('Invalid data type.')