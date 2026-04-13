import torch


def select_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        major, minor = torch.cuda.get_device_capability()
        supported_arches = set(torch.cuda.get_arch_list())
    except Exception:
        return torch.device("cpu")

    device_arch = f"sm_{major}{minor}"
    if device_arch not in supported_arches:
        print(
            f"CUDA device capability {device_arch} is not supported by the current PyTorch build; "
            "falling back to CPU."
        )
        return torch.device("cpu")

    return torch.device("cuda")
