import torch

SPEAKER_START = 49518 
SPEAKER_END = 22560 
NULL_ID_FOR_COREF = 0

def flatten(l):
    return [item for sublist in l for item in sublist]

def bucket_distance(offsets):
    """ offsets: [num spans1, num spans2] """
    # 10 semi-logscale bin: 0, 1, 2, 3, 4, (5-7)->5, (8-15)->6, (16-31)->7, (32-63)->8, (64+)->9
    logspace_distance = torch.log2(offsets.to(torch.float)).to(torch.long) + 3
    identity_mask = (offsets <= 4).to(torch.long)
    combined_distance = identity_mask * offsets + (1 - identity_mask) * logspace_distance
    combined_distance = torch.clamp(combined_distance, 0, 9)
    return combined_distance


def batch_select(tensor, idx, device=torch.device('cpu')):
    """ select on the last axis according to the index """
    assert tensor.shape[0] == idx.shape[0]  # Same size of first dim
    dim0_size, dim1_size = tensor.shape[0], tensor.shape[1]
    tensor = torch.reshape(tensor, [dim0_size * dim1_size, -1])
    idx_offset = torch.unsqueeze(torch.arange(0, dim0_size, device=device) * dim1_size, 1)
    new_idx = idx + idx_offset
    selected = tensor[new_idx]
    if tensor.shape[-1] == 1:  # If selected element is scalar, restore original dim
        selected = torch.squeeze(selected, -1)
    return selected

def add_dummy(tensor, eps=0):
    """ Prepends zeros (or a very small value if eps is True) to the first (not zeroth) dimension of tensor.
    """
    kwargs=dict(device=tensor.device, dtype=tensor.dtype)
    shape=list(tensor.shape)
    shape[1]=1
    dummy=torch.full(shape, eps, **kwargs)
    return torch.cat((dummy, tensor), dim=1)

def mask_tensor(t, mask):
    t=t+((1.0-mask.float())*-10000.0)
    t=torch.clamp(t, min=-10000.0, max=10000.0)
    return t