import sys
import numpy as np
import math as math
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import KMeans, SpectralClustering
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin, ClusterMixin


class PRI(BaseEstimator, ClusterMixin, TransformerMixin):

    def __init__(self, lambda_=2, sigma_initial=30, ayota=1, max_iter=1000, tol=5E-7, reduction_=False, n_groups=3, namedb=None, method='FP', optimization = None, t0=100, t1=1000, PC=0.4):

        self.niter = max_iter
        self.tole = tol
        self.r = reduction_
        self.toln = n_groups
        self.name = namedb
        self.method = method
        self.alpha = lambda_
        self.xsigma = sigma_initial
        self.yota = ayota
        self.optimization = optimization
        self.t0 = t0
        self.t1 = t1
        self.PC = PC

    def fit(self, X):
        self.X = X
        self.cluster_centers_, self.labels_ = self.pri_fuction(X)

    def pri_fuction(self, Xo):

        #################################################### Preparar los datos #######################################################

        if (Xo.ndim == 2):
            X = np.random.uniform(low=np.amin(
                Xo[:, 1]), high=np.amax(Xo[:, 1]), size=(Xo.shape[0], Xo.shape[1]))

        elif (X.ndim == 3):
            Xo = np.reshape(X, (np.shape(Xo)[0] * np.shape(Xo)[1], 3))
            X = np.random.randint(low=np.mean(
                Xo) - 2 * np.std(Xo), high=np.mean(Xo) + 2 * np.std(Xo), size=np.shape(Xo))

        else:
            print('El arreglo tiene un numero de dimensiones incorrecto')

        ##################################################### MAIN #####################################################################

        NX = X.shape[0]
        NXo = Xo.shape[0]
        sigma = np.mean(cdist(Xo, X))
        sigmai = sigma
        Xr = Xo
        i = 0
        stopc = 1
        D = []
        J = []
        labels = np.zeros(Xo.shape[0])

        if self.optimization == 'Adam':
            optimization_model = Adam()
        elif self.optimization == 'Nadam':
            optimization_model = Nadam()
        elif self.optimization != None:
            print('optimization model could not be found')

        while (i < self.niter and abs(stopc) > self.tole):

            # Distances

            Dx1 = cdist(X, X)
            Dx2 = cdist(Xo, Xo)
            Dx3 = cdist(X, Xo)

            # Kernels

            K1 = 1 / math.sqrt(2 * np.pi * sigma**2) * \
                np.exp(-Dx1**2 / (2 * sigma**2))
            K2 = 1 / math.sqrt(2 * np.pi * sigma**2) * \
                np.exp(-Dx2**2 / (2 * sigma**2))
            K3 = 1 / math.sqrt(2 * np.pi * sigma**2) * \
                np.exp(-Dx3**2 / (2 * sigma**2))

            # Potential of information

            V1 = 1 / (NX**2) * np.sum(K1)
            V2 = 1 / (NXo**2) * np.sum(K2)
            V3 = 1 / (NX * NXo) * np.sum(K3)

            # Compute Divergence

            D.append(2 * np.log(V3) - np.log(V1) - np.log(V2))

            # Cost Function

            J.append(-(1 - self.alpha) * np.log(V1) -
                     2 * self.alpha * np.log(V3))
            #  Stop criterion
            if i == 0:

                stopc, BestX, pos, bestJ, Patient_C, pos = self.convergence(
                    X, J[i], D[i], i,)
            else:
                stopc, BestX, pos, bestJ, Patient_C, pos = self.convergence(
                    X, J[i], D[i], i, D[i - 1], J[i - 1], BestX, bestJ, Patient_C, pos)

            Xk = X
            # Update Xk

            if self.alpha != 0:
                if self.method == 'FP':
                    c = (V3 / V1) * (NXo / NX)
                    eta = (1 - self.alpha) / self.alpha
                    num = K3@np.ones(Xo.shape)
                    X = c * eta * (K1@Xk / num) + K3@Xo / num - c * eta * (K1@np.ones(X.shape) / num) * Xk

                elif self.method == 'SGD':
                    FXk = -1 / (NX * sigma**2) * (K1@Xk - K1@np.ones(X.shape) * Xk)
                    FXo = -1 / (NXo * sigma**2) * (K3@Xo - K3@np.ones(X.shape) * Xk)
                    g = 2 * (1 - self.alpha) * FXk / V1 + \
                        2 * (self.alpha * FXo) / V3
                    if self.optimization != None:
                        X = optimization_model.step(g, X, i)

                    else:
                        X = X - self.learning_schedule(i) * g

                else:
                    print('the selected method could not be recognized')

            else:
                num = K3@np.ones(Xo.shape)
                X = (K3@Xk / num)

            # Update sigma

            sigma = (self.xsigma * sigmai) / (self.yota * i + 1)

            # Save results

            Xf = BestX

            if (i == (self.niter - 1) or abs(stopc) < self.tole):

                if (self.r == True):

                    Xr = KMeans(n_clusters=self.toln).fit(BestX)
                    Xr = Xr.cluster_centers_
                    Xf = Xr
                    sigma = np.mean(cdist(Xf, Xo))

                labels = self.Divergencecs(Xo, Xf, sigma)

            i = i + 1
        try:

            self.D = D[:pos]
            self.J = J[:pos]
        except:
            self.D = D
            self.J = J

        return Xf, labels

    def convergence(self, X, J, D, i, Da = None, Ja = None, BestX = None, bestJ = None, Patient_C = None, pos = None):

        if i > 0:
            stopc = distance.euclidean(J, Ja)
            d = distance.euclidean(D, Da)
            if (J <= bestJ):
                bestJ = J
                BestX = X
                Patient_C = 0
                pos = i

            else:
                pos = pos
                if Patient_C == int(self.PC * self.niter) or d < self.tole:
                    if d < self.tole:
                        BestX = X
                        pos = i
                    stopc = self.tole**2
                    bestJ = self.tole ** 2
                else:
                    Patient_C += 1
        else:
            stopc = 1
            pos = i
            bestJ = sys.maxsize
            BestX = X
            Patient_C = 0

        return stopc, BestX, pos, bestJ, Patient_C, pos

    def Divergencecs(self, X, Xo, sigma):
        Dx1 = cdist(Xo, X)
        Dx2 = cdist(X, X)
        Dx3 = cdist(Xo, Xo)
        K1 = 1 / math.sqrt(2 * np.pi * sigma**2) * \
            np.exp(-Dx1**2 / (2 * sigma**2))
        K2 = 1 / math.sqrt(2 * np.pi * sigma**2) * \
            np.exp(-Dx2**2 / (2 * sigma**2))
        K3 = 1 / math.sqrt(2 * np.pi * sigma**2) * \
            np.exp(-Dx3**2 / (2 * sigma**2))
        N = np.shape(X)[0]
        No = np.shape(Xo)[0]
        A = -np.log((1 / N * No) * K1)
        B = -np.log((1 / N**2) * np.ones((Xo.shape[0], X.shape[0]))@K2)
        C = -np.log((1 / No**2) * K3@np.ones((Xo.shape[0], X.shape[0])))
        labels = 2 * A - B - C
        return np.argmin(labels, axis=0)

    def predict(self, X, y=None):
        self.labels_pred = self.fit(X).labels_
        return self.labels_pred

    def results(self):

        Xor = self.cluster_centers_

        if (self.X.ndim == 2):

            fig = plt.figure()
            ax = plt.subplot()
            ax.scatter(self.X[:, 0], self.X[:, 1], c=self.labels_)
            plt.show()

            # Final result

            fig = plt.figure()
            ax = plt.subplot()
            ax.plot(self.X[:, 0], self.X[:, 1], 'x')
            ax.plot(Xor[:, 0], Xor[:, 1], 'rx')
            plt.title('Resultado final')
            # name=DB+'_'+'Mejor_resultado'+'.png'
            # plt.savefig(name)
            plt.show()

            # Divergence Graphics

            fig = plt.figure()
            plt.plot(self.D)
            plt.xlabel('Iteraciones')
            plt.ylabel('$D_{cs}$' + '(X||Xo)')
            plt.title('Divergencia')
            # name=DB+'_'+'Divergencia.png'
            # plt.savefig(name)
            plt.show()

            # Mahalanobis Graphics

            fig = plt.figure()
            plt.plot(self.J)
            plt.xlabel('Iteraciones')
            plt.ylabel('Distancia')
            plt.title('Funcion de Costo (J)')
            # name=DB+'_'+'Distancia_Mahalanobis.png'
            # plt.savefig(name)
            plt.show()

    def learning_schedule(self, t):
        return self.t0 / (t + self.t1)

    def get_params(self, deep=True):

        return {"lambda_": self.alpha, "sigma_initial": self.xsigma, "ayota": self.yota, "method": self.method, 'optimization': self.optimization, 't0': self.t0, 't1': self.t1, 'PC': self.PC}

    def set_params(self, **parameters):

        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class spectralClustering(BaseEstimator, ClusterMixin, TransformerMixin):

    def __init__(self, n_clusters=2, gamma=1, n_neighbors=10):
        self.k = n_clusters
        self.gamma = gamma
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):

        self.cluster = SpectralClustering(n_clusters = self.k)
        self.cluster.fit(X)

        return self

    def predict(self, X):

        return self.cluster.fit_predict(X)

    def get_params(self, deep=True):

        return {"n_clusters": self.k, "gamma": self.gamma, "n_neighbors": self.n_neighbors}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class MiniBatchPRI(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, lambda_=2, sigma_initial=30, ayota=1, max_epochs=1000, tol=5E-8, reduction_=False, n_groups=3, namedb=None, minibatch_size=8, optimization = None, t0 = 1, t1 = 10000, PC = 0.4):

        self.niter = max_epochs
        self.tole = tol
        self.r = reduction_
        self.toln = n_groups
        self.name = namedb
        self.minibatch_size = minibatch_size
        self.alpha = lambda_
        self.xsigma = sigma_initial
        self.yota = ayota
        self.optimization = optimization
        self.t0 = t0
        self.t1 = t1
        self.PC = PC

    def fit(self, X, y=None):
        self.X = X
        self.cluster_centers_, self.labels_ = self.pri_MiniBatch(X)

    def pri_MiniBatch(self, Xo):

        #################################################### Preparar los datos #######################################################

        if (Xo.ndim == 2):
            X = np.random.uniform(low=np.amin(
                Xo[:, 1]), high=np.amax(Xo[:, 1]), size=(Xo.shape[0], Xo.shape[1]))

        elif (X.ndim == 3):
            Xo = np.reshape(X, (np.shape(Xo)[0] * np.shape(Xo)[1], 3))
            X = np.random.randint(low=np.mean(
                Xo) - 2 * np.std(Xo), high=np.mean(Xo) + 2 * np.std(Xo), size=np.shape(Xo))

        else:
            print('El arreglo tiene un numero de dimensiones incorrecto')

        ##################################################### MAIN #####################################################################

        Xr = Xo
        t = 0
        stopc = 1
        D = []
        J = []
        labels = np.zeros(Xo.shape[0])
        Xi = X
        if self.optimization == 'Adam':
            optimization_model = Adam()
        elif self.optimization == 'Nadam':
            optimization_model = Nadam()
        elif self.optimization != None:
            print('optimization model could not be founded')

        for epoch in range(self.niter):
            if abs(stopc) < self.tole:
                break

            shuffled_indices = np.random.permutation(Xo.shape[0])
            XoS = Xo[shuffled_indices]
            NX = X.shape[0]
            NXo = XoS.shape[0]
            sigma = np.mean(cdist(XoS, X))
            sigmai = sigma
            for i in range(0, Xo.shape[0], self.minibatch_size):

                t += 1
                Xoi = XoS[i:i + self.minibatch_size]

                # Distances

                Dx1 = cdist(Xi, Xi)
                Dx2 = cdist(Xoi, Xoi)
                Dx3 = cdist(Xi, Xoi)

                # Kernels

                K1 = 1 / math.sqrt(2 * np.pi * sigma**2) * \
                    np.exp(-Dx1**2 / (2 * sigma**2))
                K2 = 1 / math.sqrt(2 * np.pi * sigma**2) * \
                    np.exp(-Dx2**2 / (2 * sigma**2))
                K3 = 1 / math.sqrt(2 * np.pi * sigma**2) * \
                    np.exp(-Dx3**2 / (2 * sigma**2))

                # Potential of information

                V1 = 1 / (NX**2) * np.sum(K1)
                V2 = 1 / (NXo**2) * np.sum(K2)
                V3 = 1 / (NX * NXo) * np.sum(K3)

                #  Stop criterion
                if t - 1 == 0:
                    stopc, BestX, pos, d, j, bestJ, Patient_C, pos = self.convergence(
                        Xi, Xo, sigma, t - 1)
                else:

                    stopc, BestX, pos, d, j, bestJ, Patient_C, pos = self.convergence(
                        Xi, Xo, sigma, t - 1, D[t - 2], J[t - 2], BestX, bestJ, Patient_C, pos)
                # Compute Divergence

                D.append(d)

                # Cost Function

                J.append(j)
                plt.ion()
                plt.plot(Xo[:, 0], Xo[:, 1], 'r*')

                Xk = Xi
                # Update Xk

                if self.alpha != 0:

                    FXk = -1 / (NX * sigma**2) * (K1@Xk - K1@np.ones(Xi.shape) * Xk)
                    FXo = -1 / (NXo * sigma**2) * (K3@Xoi - K3@np.ones(Xoi.shape) * Xk)
                    g = 2 / self.minibatch_size * (2 * (1 -
                                                        self.alpha) * FXk / V1 + 2 * (self.alpha * FXo) / V3)
                    if self.optimization != None:
                        Xi = optimization_model.step(g, Xi, t - 1)
                    else:
                        Xi = Xi - self.learning_schedule(t) * g
                        print(t)
                        plt.cla()
                        plt.plot(Xo[:, 0], Xo[:, 1], 'r*')
                        plt.plot(Xi[:, 0], Xi[:, 1], 'b*')
                        plt.pause(0.01)
                        plt.show()
                else:
                    num = K3@np.ones(Xoi.shape)
                    Xi = (K3@Xk / num)

                # Update sigma

                sigma = (self.xsigma * sigmai) / (self.yota * t + 1)

                # Save results
                Xf = BestX
                if (epoch == (self.niter - 1) or abs(stopc) < self.tole):

                    if (self.r == True):

                        Xr = KMeans(n_clusters=self.toln).fit(BestX)
                        Xr = Xr.cluster_centers_
                        Xf = Xr
                        sigma = np.mean(cdist(Xf, Xo))
                    labels = self.Divergencecs(Xo, Xf, sigma)
                    break

        try:

            self.D = D[:pos]
            self.J = J[:pos]
        except:
            self.D = D
            self.J = J

        return Xf, labels

    def convergence(self, X, Xo, sigma, i, Da = None, Ja = None, BestX = None, bestJ = None, Patient_C = None, pos = None):
        NX = X.shape[0]
        NXo = Xo.shape[0]

        Dx1 = cdist(X, X)
        Dx2 = cdist(Xo, Xo)
        Dx3 = cdist(X, Xo)

        # Kernels

        K1 = 1 / math.sqrt(2 * np.pi * sigma**2) * \
            np.exp(-Dx1**2 / (2 * sigma**2))
        K2 = 1 / math.sqrt(2 * np.pi * sigma**2) * \
            np.exp(-Dx2**2 / (2 * sigma**2))
        K3 = 1 / math.sqrt(2 * np.pi * sigma**2) * \
            np.exp(-Dx3**2 / (2 * sigma**2))

        # Potential of information

        V1 = 1 / (NX**2) * np.sum(K1)
        V2 = 1 / (NXo**2) * np.sum(K2)
        V3 = 1 / (NX * NXo) * np.sum(K3)
        D = 2 * np.log(V3) - np.log(V1) - np.log(V2)
        J = -(1 - self.alpha) * np.log(V1) - 2 * self.alpha * np.log(V3)

        if i > 0:
            stopc = distance.euclidean(J, Ja)

            if (J <= bestJ):

                bestJ = J
                BestX = X
                Patient_C = 0
                pos = i

            else:
                pos = pos
                if Patient_C == int(self.PC * self.niter) or stopc < self.tole:
                    if stopc < self.tole:
                        BestX = X
                        pos = i
                    stopc = self.tole**2
                    bestJ = self.tole ** 2
                else:
                    Patient_C += 1
        else:
            stopc = 1
            pos = i
            bestJ = sys.maxsize
            BestX = X
            Patient_C = 0

        return stopc, BestX, pos, D, J, bestJ, Patient_C, pos

    def Divergencecs(self, X, Xo, sigma):
        Dx1 = cdist(Xo, X)
        Dx2 = cdist(X, X)
        Dx3 = cdist(Xo, Xo)
        K1 = 1 / math.sqrt(2 * np.pi * sigma**2) * \
            np.exp(-Dx1**2 / (2 * sigma**2))
        K2 = 1 / math.sqrt(2 * np.pi * sigma**2) * \
            np.exp(-Dx2**2 / (2 * sigma**2))
        K3 = 1 / math.sqrt(2 * np.pi * sigma**2) * \
            np.exp(-Dx3**2 / (2 * sigma**2))
        N = np.shape(X)[0]
        No = np.shape(Xo)[0]
        A = -np.log((1 / N * No) * K1)
        B = -np.log((1 / N**2) * np.ones((Xo.shape[0], X.shape[0]))@K2)
        C = -np.log((1 / No**2) * K3@np.ones((Xo.shape[0], X.shape[0])))
        labels = 2 * A - B - C
        return np.argmin(labels, axis=0)

    def predict(self, X):
        self.labels_pred = self.fit(X).labels_
        return self.labels_pred

    def results(self):

        Xor = self.cluster_centers_

        if (self.X.ndim == 2):

            fig = plt.figure()
            ax = plt.subplot()
            ax.scatter(self.X[:, 0], self.X[:, 1], c=self.labels_)
            plt.show()

            # Final result

            fig = plt.figure()
            ax = plt.subplot()
            ax.plot(self.X[:, 0], self.X[:, 1], 'x')
            ax.plot(Xor[:, 0], Xor[:, 1], 'rx')
            plt.title('Resultado final')
            # name=DB+'_'+'Mejor_resultado'+'.png'
            # plt.savefig(name)
            plt.show()

            # Divergence Graphics

            fig = plt.figure()
            plt.plot(self.D)
            plt.xlabel('Iteraciones')
            plt.ylabel('$D_{cs}$' + '(X||Xo)')
            plt.title('Divergencia')
            # name=DB+'_'+'Divergencia.png'
            # plt.savefig(name)
            plt.show()

            # Mahalanobis Graphics

            fig = plt.figure()
            plt.plot(self.J)
            plt.xlabel('Iteraciones')
            plt.ylabel('Distancia')
            plt.title('Funcion de Costo (J)')
            # name=DB+'_'+'Distancia_Mahalanobis.png'
            # plt.savefig(name)
            plt.show()

    def get_params(self, deep=True):

        return {"lambda_": self.alpha, "sigma_initial": self.xsigma, "ayota": self.yota, 'minibatch_size': self.minibatch_size, 'optimization': self.optimization, 't0': self.t0, 't1': self.t1, 'PC': self.PC}

    def learning_schedule(self, t):
        return self.t0 / (t + self.t1)

    def set_params(self, **parameters):

        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class Adam(BaseEstimator, ClusterMixin, TransformerMixin):
    def __init__(self, alpha = 0.001, B1 = 0.9, B2 = 0.999, e = 1E-8):

        self.B1 = B1
        self.B2 = B2
        self.e = e
        self.alpha = alpha

    def step(self, g, X, i):
        if i == 0:
            self.m = 0
            self.v = 0

        self.m = self.B1 * self.m + (1 - self.B1) * g
        self.v = self.B2 * self.v + (1 - self.B2) * np.power(g, 2)

        m_hat = self.m / (1 - np.power(self.B1, i + 1))
        v_hat = self.v / (1 - np.power(self.B2, i + 1))

        X = X - self.alpha * \
            m_hat / (np.sqrt(v_hat) + self.e)
        return X

    def get_params(self, deep=True):

        return {"alpha": self.alpha, "Beta1": self.B1, "Beta2": self.B2, 'epsilon': self.e}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class Nadam(BaseEstimator, ClusterMixin, TransformerMixin):

    def __init__(self, alpha = 0.001, B1 = 0.9, B2 = 0.999, e = 1E-8):

        self.B1 = B1
        self.B2 = B2
        self.e = e
        self.alpha = alpha

    def step(self, g, X, i):
        if i == 0:
            self.m = 0
            self.v = 0

        self.m = self.B1 * self.m + (1 - self.B1) * g
        self.v = self.B2 * self.v + (1 - self.B2) * np.power(g, 2)

        m_hat = self.m / (1 - np.power(self.B1, i + 1)) + (1 - self.B1) * g /\
            (1 - np.power(self.B1, i + 1))

        v_hat = self.v / (1 - np.power(self.B2, i + 1))

        X = X - self.alpha * \
            m_hat / (np.sqrt(v_hat) + self.e)
        return X

    def get_params(self, deep=True):

        return {"alpha": self.alpha, "Beta1": self.B1, "Beta2": self.B2, 'epsilon': self.e}

    def set_params(self, **parameters):

        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


data = sio.loadmat('DB.mat')['DB'][0, 0]
happy = data['happy']
sc = SpectralClustering(n_clusters=3, n_neighbors=5, gamma = 1000)
sc.fit(happy)
labels_happy = sc.labels_
p = MiniBatchPRI(n_groups=3, lambda_=35, sigma_initial=70,
                 reduction_ = False, t0=1, t1=100000)
p.fit(happy)
p.results()


