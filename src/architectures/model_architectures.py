from sklearn.linear_model import LinearRegression


class LinearRegressor():
    def __init__(self) -> None:
        self.model = LinearRegression()

    def fit_data(self, X, Y) -> None:
        self.model.fit(X, Y)

    def forecast_data(self, X):
        Y = self.model.predict(X)
        Y[Y < 0] = 0
        return Y
