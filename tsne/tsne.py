import numpy as np
from loguru import logger


class CustomTSNE:
    def __init__(self, X: np.ndarray, no_dims: int = 2, initial_dims: int = 50,
                 perplexity: float = 30.0, max_iter: int = 1000):

        self.no_dims: int = no_dims
        self.initial_dims: int = initial_dims
        self.perplexity: float = perplexity
        self.max_iter: int = max_iter
        self.X: np.ndarray = X

    def compute_pairwise_distances(self):
        """
            Вычисляет матрицу попарных расстояний
        """

        sum_x = np.sum(np.square(self.X), axis=1)
        D = np.add(np.add(-2 * np.dot(self.X, self.X.T), sum_x).T, sum_x)
        return D

    def compute_pij(self, D, perplexity=30.0, tol=1e-5):
        """Вычисляет матрицу вероятностей P."""
        n = D.shape[0]
        P = np.zeros((n, n))
        beta = np.ones((n, 1))
        logU = np.log(perplexity)

        for i in range(n):
            betamin = -np.inf
            betamax = np.inf
            Di = np.concatenate((D[i, :i], D[i, i + 1:]))
            H, thisP = self.hbeta(Di, beta[i])

            Hdiff = H - logU
            tries = 0
            while np.abs(Hdiff) > tol and tries < 50:
                if Hdiff > 0:
                    betamin = beta[i, 0]
                    beta[i, 0] = (beta[i, 0] + betamax) / 2 if np.isfinite(betamax) else beta[i, 0] * 2
                else:
                    betamax = beta[i, 0]
                    beta[i, 0] = (beta[i, 0] + betamin) / 2 if np.isfinite(betamin) else beta[i, 0] / 2
                H, thisP = self.hbeta(Di, beta[i])
                Hdiff = H - logU
                tries += 1

            P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

        P = (P + P.T) / (2 * n)  # Симметризация и нормализация
        return P

    @staticmethod
    def hbeta(Di, beta):
        """
            Вычисляет энтропию и вероятности для данного β
        """
        P = np.exp(-Di * beta)
        sumP = np.sum(P)
        H = np.log(sumP) + beta * np.sum(Di * P) / sumP
        P = P / sumP
        return H, P

    def study(self):
        """
            Основная функция t-SNE
        """
        logger.info('start train custom t-sne')

        # Снижение размерности с помощью PCA
        self.X = self.X - np.mean(self.X, axis=0)
        cov = np.cov(self.X.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1][:self.initial_dims]
        eigvecs = eigvecs[:, idx]
        self.X = np.dot(self.X, eigvecs)

        n, d = self.X.shape
        Y = np.random.randn(n, self.no_dims)
        dY = np.zeros((n, self.no_dims))
        iY = np.zeros((n, self.no_dims))
        gains = np.ones((n, self.no_dims))

        # Вычисление матрицы вероятностей P
        D = self.compute_pairwise_distances()
        P = self.compute_pij(D=D, perplexity=self.perplexity)
        P = P * 4.0  # Усиление начальных градиентов
        P = np.maximum(P, 1e-12)

        for iter in range(self.max_iter):

            # Вычисление матрицы вероятностей Q
            sum_Y = np.sum(np.square(Y), axis=1)
            num = -2. * np.dot(Y, Y.T)
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
            num[range(n), range(n)] = 0.
            Q = num / np.sum(num)
            Q = np.maximum(Q, 1e-12)

            # Обновление градиента
            PQ = P - Q
            for i in range(n):
                dY[i, :] = np.sum((PQ[:, i] * num[:, i])[:, np.newaxis] * (Y[i, :] - Y), axis=0)

            # Адаптивная скорость обучения
            gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + gains * 0.8 * ((dY > 0.) == (iY > 0.))
            gains[gains < 0.01] = 0.01
            iY = 0.8 * iY - gains * dY
            Y += iY
            Y -= np.mean(Y, axis=0)

            # Снижение влияния P после первых 100 итераций
            if iter == 100:
                P = P / 4.0

            # Вывод ошибки каждые 100 итераций
            if (iter + 1) % 100 == 0:
                C = np.sum(P * np.log(P / Q))
                logger.debug(f"Iteration {iter + 1}: error = {C}")

        logger.info('finish train custom t-sne')
        return Y
