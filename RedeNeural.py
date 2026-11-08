import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preparar_dados(test_size: float = 0.20):
    iris = load_iris()
    X = iris.data.astype(np.float64)
    y = iris.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train_one_hot = np.eye(3, dtype=np.float64)[y_train]
    y_test_one_hot = np.eye(3, dtype=np.float64)[y_test]

    return X_train, X_test, y_train, y_test, y_train_one_hot, y_test_one_hot

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)

def derivada_relu(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(np.float64)

def softmax(z: np.ndarray) -> np.ndarray:
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def entropia_cruzada(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(-np.sum(y_true * np.log(y_pred), axis=1)))

class RedeNeuralIris:
    def __init__(self, tamanho_entrada: int = 4, tamanho_oculto: int = 8, tamanho_saida: int = 3):
        self.W1 = np.random.randn(tamanho_entrada, tamanho_oculto) * 0.01
        self.b1 = np.zeros((1, tamanho_oculto), dtype=np.float64)

        self.W2 = np.random.randn(tamanho_oculto, tamanho_saida) * 0.01
        self.b2 = np.zeros((1, tamanho_saida), dtype=np.float64)

    def forward(self, X: np.ndarray):
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = softmax(z2)
        return a1, a2, z1

    def treinar(self, X: np.ndarray, y_one_hot: np.ndarray, epocas: int = 1000, taxa_aprendizado: float = 0.1):
        N = X.shape[0]

        for epoca in range(1, epocas + 1):
            a1, a2, z1 = self.forward(X)

            dz2 = (a2 - y_one_hot) / N
            dW2 = a1.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)

            da1 = dz2 @ self.W2.T
            dz1 = da1 * derivada_relu(z1)
            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            self.W2 -= taxa_aprendizado * dW2
            self.b2 -= taxa_aprendizado * db2
            self.W1 -= taxa_aprendizado * dW1
            self.b1 -= taxa_aprendizado * db1

def main():
    X_train, X_test, y_train, y_test, y_train_one_hot, y_test_one_hot = preparar_dados()

    modelo = RedeNeuralIris(tamanho_entrada=4, tamanho_oculto=8, tamanho_saida=3)
    modelo.treinar(X_train, y_train_one_hot, epocas=1000, taxa_aprendizado=0.1)

    _, probabilidades_teste, _ = modelo.forward(X_test)
    previsoes_teste = np.argmax(probabilidades_teste, axis=1)

    acuracia = np.mean(previsoes_teste == y_test)
    print(f"Acurácia final no conjunto de teste: {acuracia * 100:.2f}%")

if __name__ == "__main__":
    main()