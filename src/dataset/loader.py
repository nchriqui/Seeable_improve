import torch
from torch.utils.data import DataLoader
from src.dataset.collator import collate_fn

class Loader:
    def __init__(self, dataset, batch_size, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, pin_memory=False, drop_last=True, timeout=0, worker_init_fn=0, persistent_workers=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.persistent_workers = persistent_workers

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            sampler=self.sampler,
            batch_sampler=self.batch_sampler,            
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            timeout=self.timeout,
            worker_init_fn=self.worker_init_fn,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn
        )

    # def run(self):
    #     return DataLoader(
    #         dataset=self.dataset,
    #         batch_size=self.batch_size,
    #         shuffle=self.shuffle,
    #         sampler=self.sampler,
    #         batch_sampler=self.batch_sampler,            
    #         num_workers=self.num_workers,
    #         pin_memory=self.pin_memory,
    #         drop_last=self.drop_last,
    #         timeout=self.timeout,
    #         worker_init_fn=self.worker_init_fn,
    #         persistent_workers=self.persistent_workers,
    #         collate_fn=collate_fn
    #     )
    
    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)
