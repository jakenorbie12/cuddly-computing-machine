import math

from sklearn.metrics import mean_squared_log_error


def evaluate_error(Y_true, Y_pred):
    return math.sqrt(mean_squared_log_error(Y_true, Y_pred))
