import torch


def collate_fn(batch):
    """collate function for torch.utils.data.DataLoader
    Input:
        batch: list of dict
    Output:
        images: torch.Tensor; shape (batch_size, ...)
        target: torch.Tensor; shape (batch_size,    )
    """
    images = [b["image"] for b in batch]
    targets = [b["label"] for b in batch]
    # stack images
    images = torch.stack(images)
    targets = torch.tensor(targets)
    #
    L = ["image", "label"]
    samples = [{k: v for k, v in b.items() if k not in L} for b in batch]
    return images, targets, samples