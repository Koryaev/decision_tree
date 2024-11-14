import sys

import pandas as pd
from typing import List
from loguru import logger


def load_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logger.info('csv file loaded')
    except FileNotFoundError:
        logger.error(f'incorrect path: "{path}"')
        sys.exit(1)

    return df


def percent_similarity(correct_list: List[int] , predicted_list: List[int]) -> float:
    if len(correct_list) != len(predicted_list):
        logger.error(f'lens of predicted and correct answers is different! '
                     f'Predicted count" {len(predicted_list)}, correct: {len(correct_list)}')
        sys.exit(1)

    correct: int = 0

    for i in range(len(correct_list)):
        if correct_list[i] == predicted_list[i]:
            correct += 1

    return (correct / len(correct_list)) * 100
