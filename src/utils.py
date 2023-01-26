def inverse_transform(source_tensor):
    """
    source_tensor : B, C, H, W

    target_tensor : B, H, W, C

    :param source_tensor: torch.Tensor()
    :return: target_tensor: torch.Tensor()
    """
    assert source_tensor.ndim == 4
    source_tensor = source_tensor.detach().cpu()
    target_tensor = source_tensor.permute(0, 2, 3, 1)
    return target_tensor