from __future__ import annotations
from core.models.pinn import PINN

import copy
from typing import cast

from examples.any.model import AnyPINN
from core.utils.convert import convert


class Node:
    def __init__(self, pinn: PINN | None, total_epoch: int, id: int, 
                 possibles_params: list[tuple[str, str]], current_steps: int, 
                 factor: str, optimizer: str, parent: Node | None = None, 
                 epoch: int = 0):
        
        self.pinn = copy.deepcopy(pinn) if pinn is not None else None
        self.id = id
        self.current_steps = current_steps
        self.total_epoch = total_epoch
        
        self.possibles_params = copy.deepcopy(possibles_params)
        self.factor = factor
        self.optimizer = optimizer
        self.epoch = epoch

        self.parent: Node | None = parent

    def set_model(self):
        if self.parent is not None and self.parent.pinn is not None:
            self.pinn = cast(AnyPINN, convert(copy.deepcopy(self.parent.pinn), self.parent.factor, self.factor))

    def get_word(self) -> list[list[str]]:
        match self.parent:
            case None:
                return [[self.factor, self.optimizer]]
            case Node() if self.parent.id <= 0:
                return [[self.factor, self.optimizer]]
            case Node():
                return self.parent.get_word() + [[self.factor, self.optimizer]]


    def __str__(self):
        return f"Node(id={self.id}, factor={self.factor}, optimizer={self.optimizer}, epoch={self.epoch}, current_steps={self.current_steps}, total_epoch={self.total_epoch})"

