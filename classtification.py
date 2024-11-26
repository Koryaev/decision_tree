import numpy as np
import plotly.express as px
from sklearn.datasets import load_digits  # MNIST
from sklearn.manifold import TSNE
from loguru import logger
from tsne.tsne import CustomTSNE


def train_original_tsne(x: np.ndarray) -> np.ndarray:
    """
        n_components: Размерность вложенного пространства.
        
        perplexity:  Перплексия связана с количеством ближайших соседей,
         котороые используется в других алгоритмах обучения на множествах.
         
        early_exaggeration: Определяет, насколько плотными будут естественные кластеры исходного 
        пространства во вложенном пространстве и сколько места будет между ними.
        
        learning_rate: Скорость обучения для t-SNE обычно находится в диапазоне [10.0, 1000.0].
         Если скорость обучения слишком высока, данные могут выглядеть как "шар", 
         в котором любая точка приблизительно равноудалена от ближайших соседей. 
         Если скорость обучения слишком низкая, большинство точек могут быть похожими на сжатое плотное облако 
         с незначительным количеством разбросов.
        
        n_iter: Максимальное количество итераций для оптимизации. Должно быть не менее 250.
        
        n_iter_without_progress: Максимальное количество итераций без прогресса перед прекращением оптимизации,
        используется после 250 начальных итераций с ранним преувеличением.
         
        min_grad_norm: Если норма градиента ниже этого порога, оптимизация будет остановлена.
        
        metric: Метрика, используемая при расчете расстояния между экземплярами в массиве признаков.
        
        method: По умолчанию алгоритм вычисления градиента использует аппроксимацию Барнса-Хата, 
        работающую в течение времени O(NlogN). метод='exact' будет работать по более медленному,
         но точному алгоритму за время O(N^2). Следует использовать точный алгоритм, 
         когда количество ошибок ближайших соседей должно быть ниже 3%.
         
        angle: Используется только если метод='barnes_hut' 
        Это компромисс между скоростью и точностью в случае T-SNE с применением алгоритма Барнса-Хата.
        
        n_jobs: Количество параллельных заданий для поиска соседей. -1 означает использование всех процессоров.
    
    """
    logger.info('start train original t-sne')
    embed = TSNE(
        n_components=2,
        perplexity=10,
        early_exaggeration=12,
        learning_rate=200,
        n_iter=5000,
        n_iter_without_progress=300,
        min_grad_norm=0.0000001,
        metric='euclidean',
        init='random',
        verbose=0,
        random_state=42,
        method='barnes_hut',
        angle=0.5,
        n_jobs=-1,
    )
    res = embed.fit_transform(x)
    logger.info('finish train original t-sne')
    return res


def visualize_results(x: np.ndarray, y: np.ndarray, learning_type: str = 'original') -> None:
    logger.info(f'start visualizing {learning_type} t-sne')
    figure = px.scatter(
        None,
        x=x[:, 0], y=x[:, 1],
        labels={
            "x": "Dimension 1",
            "y": "Dimension 2",
        },
        opacity=1,
        color=y.astype(str),
    )

    # Изменение цвета фона графика
    figure.update_layout(dict(plot_bgcolor='white'))

    # Обновление линий осей
    figure.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                        zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                        showline=True, linewidth=1, linecolor='black')

    figure.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                        zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                        showline=True, linewidth=1, linecolor='black')

    figure.update_layout(title_text="t-SNE")

    figure.update_traces(marker=dict(size=3))  # Обновление размера маркера
    logger.info(f'finish visualizing {learning_type} t-sne')
    figure.show()


def main():
    logger.info('starting program')
    # Загрузка массивов, содержащих данные образцов цифр (64 пикселя на изображение) и их истинных меток
    x, y = load_digits(return_X_y=True)

    trained_original_tsne_x = train_original_tsne(x)
    visualize_results(trained_original_tsne_x, y)

    custom_tsne = CustomTSNE(
        x, no_dims=2, initial_dims=4, perplexity=10.0, max_iter=5000
    )
    custom_tsne_trained_x = custom_tsne.study()
    visualize_results(custom_tsne_trained_x, y, learning_type='custom')


if __name__ == '__main__':
    main()
