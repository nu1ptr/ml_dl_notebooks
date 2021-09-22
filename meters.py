import matplotlib.pyplot as plt
import torch
import time

class LossMeter():
    """ Keep track of losses.
    """
    def __init__(self, num_epochs, num_batches, batch_size, num_samples):
        self.num_epochs = num_epochs
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.num_samples = num_samples

        self.epoch = 1
        self.batch = 1
        self.total_batches = 0
        self.running_loss = 0.

    def step(self, cur_loss, cur_batch, cur_epoch):
        self.total_batches += 1
        self.epoch = cur_epoch
        self.batch = cur_batch + 1 # cur_batch is assumed to be 0-indexed
        self.running_loss += cur_loss

        return

    def print(self):
        print(f'Epochs: [{self.epoch}/{self.num_epochs}]\tBatches: [{self.batch}/{self.num_batches}]\tSamples: [{self.batch*self.batch_size}/{self.num_samples}]\tRunning Loss: {self.running_loss / self.batch}')
        return

    def reset(self):
        self.running_loss = 0.
        return

class HWMeter():
    """ Keep track of some basic hardware stats.
        Mainly memory usage for RAM/VRAM
    """

    def __init__(self, device='cuda:0'):
        self.device = device
        self.running_batch_time = 0.
        self.batch = 1
        self.last_time = time.time()
        return

    def print(self):
        cache_mem_reserved = torch.cuda.memory_reserved(self.device) / (2**10)
        cache_max_mem_reserved = torch.cuda.max_memory_reserved(self.device) / (2**10)
        mem_alloc = torch.cuda.memory_allocated(self.device) / (2**10)
        mem_max_alloc = torch.cuda.max_memory_allocated(self.device) / (2**10)

        #print(f'GPU Memory Cache Usage: [{cache_mem_reserved}/{cache_max_mem_reserved}]\tGPU Memory [{mem_alloc}/{mem_max_alloc}]\tRunning Batch Time: {self.running_batch_time / self.batch}s')
        print(f'Running Batch Time: {self.running_batch_time / self.batch}')
        return

    def step(self, cur_batch):
        now_time = time.time()

        self.batch = cur_batch + 1
        self.running_batch_time += now_time - self.last_time
        self.last_time = now_time
        return

    def reset(self):
        self.running_batch_time = 0.
        self.batch = 1
        return
