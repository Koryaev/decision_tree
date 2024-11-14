import pandas as pd
from loguru import logger

from utils import load_csv, percent_similarity
from sklearn.model_selection import train_test_split
from settings import settings

from CART.tree import DecisionTree


if __name__ == '__main__':
    logger.info('program start')

    df: pd.DataFrame = load_csv(settings.path_to_file)

    data = df.drop('Purchased', axis=1).to_numpy()
    target = df['Purchased'].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        data,
        target,
        train_size=settings.train_size,
        random_state=settings.random_state,
    )

    logger.info('data prepared for learning')

    tree = DecisionTree(
        depth=settings.depth,
        samples=settings.samples,
    )
    tree.process_tree(
        data=x_train,
        target=y_train,
        node=tree.root_node
    )

    logger.info('tree learned')

    percent_of_good_predicted = percent_similarity(
        correct_list=y_test.tolist(),
        predicted_list=[tree.root_node.find_decision(i) for i in x_test.tolist()]
    )
    logger.info(f'Predicted correctly: {percent_of_good_predicted:.2f}%')
