from typing import Optional
import numpy as np


class Node:
    def __init__(self):
        self.threshold: float = 0               # порог для разбиения
        self.feature: int = 0                   # индекс признака для разбиения
        self.left_node: Optional[Node] = None
        self.right_node: Optional[Node] = None
        self.is_leaf: bool = False
        self.decision: Optional[int] = None

    def find_decision(self, element: np.ndarray) -> int:
        if self.is_leaf:
            return self.decision

        elif element[self.feature] < self.threshold:
            return self.left_node.find_decision(element)

        else:
            return self.right_node.find_decision(element)
