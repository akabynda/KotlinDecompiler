import torch


def model_batch_size(model: torch.nn.Module, scale: float = 1.0) -> int:
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    vram = props.total_memory if props else 8 << 30
    size = sum(p.numel() * p.element_size() for p in model.parameters())
    return max(1, int(vram / size * scale))
