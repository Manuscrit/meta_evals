import torch


def check_for_mps():
    built = torch.backends.mps.is_built()
    available = torch.backends.mps.is_available()
    print(f"Built: {built}, Available: {available}")
    return built and available
