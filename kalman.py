import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import statsmodels.api as sm

class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

if __name__ == '__main__':

    fs,signal = wavfile.read('normal.wav')
    noise = np.random.normal(0, 100, len(signal))
    data = signal + noise

    # Estimate AR model for data and noise
    rho_signal, sigma_signal = sm.regression.yule_walker(signal, order=1, method="mle")
    rho_noise, sigma_noise = sm.regression.yule_walker(noise, order=1, method="mle")

    F = np.array([[rho_signal[0], 0], [0, rho_noise[0]]])

    H = np.array([1, 0]).reshape(1, 2)

    # The covariance of the observation noise
    R = np.array([10000]).reshape(1, 1)

    # The covariance of the process noise
    Q = np.dot(np.dot(np.array([[1, 0], [0, 1]]), 
        np.array([[sigma_signal*sigma_signal, 0], [0, sigma_noise*sigma_noise]])), 
            np.array([[1, 0], [0, 1]]))

    kf = KalmanFilter(F = F, H = H, Q= Q, R = R)
    predictions = []

    for z in data:
        predictions.append(np.dot(H, kf.predict())[0])
        kf.update(z)

    result = np.array(predictions).reshape(len(data),)

    wavfile.write('result.wav', fs, np.int16(result))
    wavfile.write('noise.wav', fs, np.int16(data))

    plt.plot(range(len(signal)), data, label = 'Data with Noise')
    plt.plot(range(len(result)), result, label='Prediction')
    plt.legend()
    plt.show()