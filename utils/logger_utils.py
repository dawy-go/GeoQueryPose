import logging
import os

import numpy as np
import torch
from tensorboardX import SummaryWriter


def get_logger(level_print, level_save, path_file, name_logger="logger"):
    logger = logging.getLogger(name_logger)
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler_file = logging.FileHandler(path_file)
    handler_file.setLevel(level_save)
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    handler_view = logging.StreamHandler()
    handler_view.setFormatter(formatter)
    handler_view.setLevel(level_print)
    logger.addHandler(handler_view)
    return logger

class tools_writer:
    def __init__(self, dir_project, num_counter, get_sum, start_step=0):
        if not os.path.isdir(dir_project):
            os.makedirs(dir_project)
        if get_sum:
            writer = SummaryWriter(dir_project)
        else:
            writer = None
        self.writer = writer
        self.num_counter = num_counter
        self.list_couter = [start_step] * num_counter

    def update_scalar(self, list_name, list_value, index_counter, prefix):
        for name, value in zip(list_name, list_value):
            self.writer.add_scalar(prefix + name, float(value), self.list_couter[index_counter])

        self.list_couter[index_counter] += 1

    def update_graph(self, model, input):
        self.writer.add_graph(model, input)

    def refresh(self):
        for i in range(self.num_counter):
            self.list_couter[i] = 0

class LogBuffer:
    def __init__(self):
        self._history = {}
        self._output = {}

    def update(self, values):
        for key, value in values.items():
            if torch.is_tensor(value):
                value = value.detach().item()
            self._history.setdefault(key, []).append(float(value))

    def average(self, n=0):
        self._output = {}
        for key, values in self._history.items():
            if not values:
                continue
            window = values[-n:] if n and n > 0 else values
            self._output[key] = float(np.mean(window))
        return self._output

    @property
    def avg(self):
        return {
            key: float(np.mean(values))
            for key, values in self._history.items()
            if values
        }

    def clear(self):
        self._history.clear()
        self._output.clear()
