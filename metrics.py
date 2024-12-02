def calculate_metrics(y_true, y_pred):
    """
    Функция для расчета F-меры, точности, полноты и выпадов.

    Параметры:
    y_true -- список истинных значений (0 или 1)
    y_pred -- список предсказанных значений (0 или 1)

    Возвращает:
    словарь с метриками
    """
    # Вычисляем основные значения
    tp = sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true)))  # Истинно положительные
    tn = sum((y_true[i] == 0 and y_pred[i] == 0) for i in range(len(y_true)))  # Истинно отрицательные
    fp = sum((y_true[i] == 0 and y_pred[i] == 1) for i in range(len(y_true)))  # Ложно положительные
    fn = sum((y_true[i] == 1 and y_pred[i] == 0) for i in range(len(y_true)))  # Ложно отрицательные

    # Точность (Precision)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    precision = round(float(precision), 2)

    # Полнота (Recall)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    recall = round(float(recall), 2)

    # Выпады (Fall-out)
    fall_out = fp / (fp + tn) if (fp + tn) > 0 else 0
    fall_out = round(float(fall_out), 2)

    # F-мера (F1-Score)
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f1_score = round(float(f1_score), 2)

    # Возвращаем все метрики в виде словаря
    return {
        "precision": precision,
        "recall": recall,
        "fall_out": fall_out,
        "f1_score": f1_score
    }


    # Пример использования
if __name__ == "__main__":
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]  # Истинные значения
    y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]  # Предсказанные значения

    metrics = calculate_metrics(y_true, y_pred)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
