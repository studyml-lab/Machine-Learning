from sklearn.metrics import mean_squared_error, explained_variance_score

y = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

mean_squared_error(y, y_pred)
explained_variance_score(y, y_pred)


y = [[0.5, 1],[-1, 1],[7, -6]]
y_pred = [[0, 2],[-1, 2],[8, -5]]
mean_squared_error(y, y_pred)  

