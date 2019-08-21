import torch
def get_current_device():
    use_coda = torch.cuda.is_available()
    device = torch.device('cuda' if use_coda else 'cpu')
    return device