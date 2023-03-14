# import required library

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import Union
import sklearn.linear_model


# %matplotlib ipympl


class HyperkneeFinder:
    """
    hyperKnee point finder.
    TIt's about a tool for optimizing two inter-dependent parameters
    """

    def __init__(self, data_x: Union[list, np.ndarray], data_y: Union[list, np.ndarray], data_z: Union[list, np.ndarray]):
        if len(data_x) != len(data_y) or len(data_x) != len(data_z) or len(data_y) != len(data_z):
            raise ValueError("Input arrays must be of the same length.")
            
        self.X = data_x
        self.Y = data_y
        self.Z = data_z
     
    #         self.xp = np.tile(np.linspace(1, 5, 61), (61, 1))
    #         self.yp = np.tile(np.linspace(6, 10, 61), (61, 1)).T
    # #         self.zp = self.factor_x * self.xp + self.factor_y * self.yp + self.new_intercept

    def plot_data(self):

        # fig = plt.figure(figsize=(7, 7))
        # ax = plt.axes(projection='3d')
        XX, YY = np.meshgrid(self.X, self.Y)

        # surf = ax.plot_surface(XX, YY, self.Z, cmap=cm.coolwarm,
        #                        linewidth=1, antialiased=True)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.show()
        return XX, YY

    def reshape_data(self):
        XX,YY = HyperkneeFinder.plot_data(self)

        X_train = np.array(
            [[XX[i, j], YY[i, j]] for i in range(len(self.X)) for j in range(len(self.Y))]).flatten().reshape(
            (len(self.X) * len(self.Y), 2))

        y_train = self.Z.flatten()

        return X_train, y_train

    def fit_model(self):
        X_train,y_train = HyperkneeFinder.reshape_data(self)
        #         self.X_train, self.y_train = self.reshape_data()
        model = sklearn.linear_model.LinearRegression()
        model = model.fit(X_train, y_train)
        return model

    def plot_fitted_plane(self):
        model = HyperkneeFinder.fit_model(self)
        XX,YY = HyperkneeFinder.plot_data(self)
        if model is None:
            print('Fit a model first.')
            return

        # fig = plt.figure(figsize=(7, 7))
        # ax = fig.add_subplot(111, projection='3d')
        # surf = ax.plot_surface(XX, YY, self.Z, cmap=cm.coolwarm,
        #                        linewidth=1, antialiased=True, alpha=0.8)
        coefs = model.coef_
        intercept = model.intercept_
        xs = np.tile(np.linspace(self.X[0], self.X[-1], 61), (61, 1))
        ys = np.tile(np.linspace(self.Y[0], self.Y[-1], 61), (61, 1)).T
        zs = xs * coefs[0] + ys * coefs[1] + intercept
        # print("Equation: z = {:.2f} + {:.2f}x + {:.2f}y".format(intercept, coefs[0], coefs[1]))
        # ax.plot_surface(xs, ys, zs, alpha=0.5)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.show()

        return

    def translate_plane(self):
        model = HyperkneeFinder.fit_model(self)

        v_n = np.array([model.coef_[0], model.coef_[1], -1])
        p0 = [self.X[0], self.Y[0], self.Z[0, 0]]
        ps_intercept = np.sum(v_n * p0)
        factor_x = model.coef_[0]
        factor_y = model.coef_[1]
        new_intercept = -ps_intercept


        # print(
        #     "Equation of the plane in normal form: {:.2f}x + {:.2f}y + {:.2f} = z".format(factor_x, factor_y,
        #                                                                                   new_intercept))

        return factor_x, factor_y, new_intercept,v_n

    def visualise1(self):
        XX,YY = HyperkneeFinder.plot_data(self)
        factor_x, factor_y, new_intercept,v_n = HyperkneeFinder.translate_plane(self)

        xp = np.tile(np.linspace(1, 5, 61), (61, 1))
        yp = np.tile(np.linspace(6, 10, 61), (61, 1)).T

        zp = factor_x * xp + factor_y * yp + new_intercept
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # surf = ax.plot_surface(XX, YY, self.Z, cmap=cm.coolwarm, linewidth=1, antialiased=True)
        # ax.set_xlabel("X1")
        # ax.set_ylabel("X2")
        # ax.set_zlabel("y")
        #
        # ax.plot_surface(xp, yp, zp, alpha=0.5)
        # plt.show()

        return xp, yp, zp

    # now that we have the plane, we must calculate the distances from the points to the plane
    # dist_1 = X_train * plane_pars[:2]
    # dist_2 = y_train * plane_pars[2]
    # dist_comb = np.concatenate((dist_1, np.expand_dims(dist_2, axis=1)), axis=1)
    # dist_tot = np.sum(np.abs(dist_comb), axis=1)

    def cal_distance(self):
        factor_x, factor_y, new_intercept, v_n = HyperkneeFinder.translate_plane(self)
        X_train, y_train = HyperkneeFinder.reshape_data(self)


        dist_1 = X_train * v_n[:2]
        dist_2 = y_train * v_n[2]

        dist_comb = np.concatenate((dist_1, np.expand_dims(dist_2, axis=1)), axis=1)
        dist_tot = np.abs(np.sum(dist_comb, axis=1) + new_intercept)

        return dist_tot

    def max_dist_from_plane(self):
        # self.dist_tot = self.cal_distance()
        dist_tot = HyperkneeFinder.cal_distance(self)

        knee_point_at = np.argmax(dist_tot)

        return knee_point_at

    def hyperkneepoint(self):
        knee_point_at = HyperkneeFinder.max_dist_from_plane(self)
        X_train, y_train = HyperkneeFinder.reshape_data(self)

        print(f"hyper-knee  at {X_train[knee_point_at]}")

#         return knee_point_at

    def visualise_hyperknee(self):
        knee_point_at = HyperkneeFinder.max_dist_from_plane(self)
        XX, YY = HyperkneeFinder.plot_data(self)
#         knee_point_at = HyperkneeFinder.hyperkneepoint(self)
        X_train, y_train = HyperkneeFinder.reshape_data(self)
        factor_x, factor_y, new_intercept, v_n = HyperkneeFinder.translate_plane(self)
        xp,yp,zp = HyperkneeFinder.visualise1(self)

        #
        # self.xp = np.tile(np.linspace(1, 5, 61), (61, 1))
        # self.yp = np.tile(np.linspace(6, 10, 61), (61, 1)).T
        # self.zp = factor_x * xp + factor_y * yp + new_intercept
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(XX, YY, self.Z, linewidth=1, antialiased=True, alpha=0.5)

        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("y")

        ax.scatter(*(X_train[knee_point_at]),y_train[knee_point_at], c='b', s=30,
                   label='knee point')
        ax.plot_surface(xp, yp, zp, alpha=0.5)
        plt.legend()
        plt.show()
