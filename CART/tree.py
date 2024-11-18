from typing import Optional, Tuple

import numpy as np

from CART.node import Node


class DecisionTree:
    def __init__(self, depth, samples):
        self.max_depth: int = depth
        self.min_samples: int = samples
        self.root_node: Node = Node()

    def count_gini_split(self, left_target, right_target) -> float:
        """
            считаем коэф gini split из gini из левой и правой
            чем мееньше, тем лучше разделение
        """
        total_size = left_target.size + right_target.size
        left_part = left_target.size / total_size
        right_part = right_target.size/total_size

        gini_split = left_part * self.count_gini(left_target) + right_part * self.count_gini(right_target)
        return gini_split

    def optimize(self, data: np.ndarray, target: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        best_threshold: Optional[float] = None
        best_feature: Optional[int] = None
        min_gini: float = np.inf
        transported_data: np.ndarray = data.transpose()

        for i, batch in enumerate(transported_data):
            for unique_el in np.unique(batch):
                left_target, right_target, _, _ = self.split(
                    data=data,
                    target=target,
                    threshold=unique_el,
                    feature=i,
                )
                gini_split = self.count_gini_split(
                    left_target=left_target,
                    right_target=right_target,
                )
                if gini_split < min_gini:
                    min_gini = gini_split
                    best_threshold = unique_el
                    best_feature = i

        return best_feature, best_threshold

    @staticmethod
    def split(data: np.ndarray, target: np.ndarray,
              threshold: float, feature: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
            разделяем массив на 2 части по которой лучше разделить
            Выбираем характеристику и порог
        """
        left_target = []
        right_target = []
        left_data = []
        right_data = []

        for i, el in enumerate(data):
            if el[feature] < threshold:
                left_target.append(target[i])
                left_data.append((data[i]))
            else:
                right_target.append(target[i])
                right_data.append(data[i])

        return np.array(left_target), np.array(right_target), np.array(left_data), np.array(right_data)

    def need_to_stop(self, target: np.ndarray, depth: int) -> bool:
        if np.unique(target).size == 1 or depth == self.max_depth or target.size < self.min_samples:
            return True
        return False

    def process_tree(self, data: np.ndarray, target: np.ndarray, node: Node, depth: int = 0) -> None:
        if self.need_to_stop(target=target, depth=depth):
            classes, counts = np.unique(target, return_counts=True)  # сколько осталось одинаковых классов в target
            decision = classes[np.where(counts == max(counts))][0]
            node.is_leaf = True
            node.decision = decision
            return

        feature, threshold = self.optimize(data, target)
        left_tar, right_tar, left_data, right_data = self.split(data, target, threshold, feature)

        node.left_node = Node()
        node.right_node = Node()
        node.threshold = threshold
        node.feature = feature
        self.process_tree(
            data=left_data,
            target=left_tar,
            node=node.left_node,
            depth=depth + 1
        )
        self.process_tree(
            data=right_data,
            target=right_tar,
            node=node.right_node,
            depth=depth + 1
        )

    @staticmethod
    def count_gini(data) -> float:
        _, counts_classes = np.unique(data, return_counts=True)
        squared_probabilities = np.square(counts_classes / data.size)
        gini = 1 - sum(squared_probabilities)

        return gini
