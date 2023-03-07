# import required library

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn.linear_model
# %matplotlib ipympl


class HyperkneeFinder:
    """
    hyperKnee point finder.
    TIt's about a tool for optimizing two inter-dependent parameters
    """

    def __init__(self, start_x, end_x, step_x, start_y, end_y, step_y):
        
        self.X = np.arange(start_x, end_x, step_x)
        self.Y = np.arange(start_y, end_y, step_y)
        self.Z = np.zeros((len(self.X), len(self.Y)))
        for i in range(len(self.X)):
            for j in range(len(self.Y)):
                self.Z[i, j] = np.exp(-(self.X[i])) + np.exp(-(self.Y[j] - 5)) + np.random.rand() / 25
        
        self.xp = np.tile(np.linspace(1, 5, 61), (61, 1))
        self.yp = np.tile(np.linspace(6, 10, 61), (61, 1)).T

    def plot_data(self):

        fig = plt.figure(figsize=(7, 7))
        ax = plt.axes(projection='3d')
        self.XX, self.YY = np.meshgrid(self.X, self.Y)

        surf = ax.plot_surface(self.XX, self.YY, self.Z, cmap=cm.coolwarm,
                               linewidth=1, antialiased=True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def reshape_data(self):
        self.X_train = np.array(
            [[self.XX[i, j], self.YY[i, j]] for i in range(len(self.X)) for j in range(len(self.Y))]).flatten().reshape(
            (len(self.X) * len(self.Y), 2))

        self.y_train = self.Z.flatten()

        return self.X_train, self.y_train
    
    
    

    def fit_model(self):
#         self.X_train, self.y_train = self.reshape_data()
        self.model = sklearn.linear_model.LinearRegression()
        return self.model.fit(self.X_train, self.y_train)

    def plot_fitted_plane(self):
        if self.model is None:
            print('Fit a model first.')
            return

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(self.XX, self.YY, self.Z, cmap=cm.coolwarm,
                               linewidth=1, antialiased=True, alpha=0.8)
        coefs = self.model.coef_
        intercept = self.model.intercept_
        xs = np.tile(np.linspace(self.X[0], self.X[-1], 61), (61, 1))
        ys = np.tile(np.linspace(self.Y[0], self.Y[-1], 61), (61, 1)).T
        zs = xs * coefs[0] + ys * coefs[1] + intercept
        print("Equation: z = {:.2f} + {:.2f}x + {:.2f}y".format(intercept, coefs[0], coefs[1]))
        ax.plot_surface(xs, ys, zs, alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    def translate_plane(self):

        p0 = [self.X[0], self.Y[0], self.Z[0, 0]]
        self.v_n = np.array([self.model.coef_[0], self.model.coef_[1], -1])
        ps_intercept = np.sum(self.v_n * p0)
        self.factor_x = self.model.coef_[0]
        self.factor_y = self.model.coef_[1]
        self.new_intercept = -ps_intercept

        print(
            "Equation of the plane in normal form: {:.2f}x + {:.2f}y + {:.2f} = z".format(self.factor_x, self.factor_y,
                                                                                          self.new_intercept))

        return self.factor_x, self.factor_y, self.new_intercept

    def visualise1(self):
        self.xp = np.tile(np.linspace(1, 5, 61), (61, 1))
        self.yp = np.tile(np.linspace(6, 10, 61), (61, 1)).T

        self.zp = self.factor_x * self.xp + self.factor_y * self.yp + self.new_intercept
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(self.XX, self.YY, self.Z, cmap=cm.coolwarm, linewidth=1, antialiased=True)
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("y")

        ax.plot_surface(self.xp, self.yp, self.zp, alpha=0.5)
        plt.show()

    # now that we have the plane, we must calculate the distances from the points to the plane
    # dist_1 = X_train * plane_pars[:2]
    # dist_2 = y_train * plane_pars[2]
    # dist_comb = np.concatenate((dist_1, np.expand_dims(dist_2, axis=1)), axis=1)
    # dist_tot = np.sum(np.abs(dist_comb), axis=1)

    def cal_distance(self):

        dist_1 = self.X_train * self.v_n[:2]
        dist_2 = self.y_train * self.v_n[2]

        dist_comb = np.concatenate((dist_1, np.expand_dims(dist_2, axis=1)), axis=1)
        dist_tot = np.abs(np.sum(dist_comb, axis=1) + self.new_intercept)

        return dist_tot

    def max_dist_from_plane(self):
        self.dist_tot = self.cal_distance()

        self.knee_point_at = np.argmax(self.dist_tot)

        return self.knee_point_at

    def hyperkneepoint(self):

        print(f"hyper-knee  at {self.X_train[self.knee_point_at]}")

    def visualise_hyperknee(self):
        self.xp = np.tile(np.linspace(1, 5, 61), (61, 1))
        self.yp = np.tile(np.linspace(6, 10, 61), (61, 1)).T
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(self.XX, self.YY, self.Z, linewidth=1, antialiased=True, alpha=0.5)

        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("y")

        ax.scatter(*self.X_train[self.knee_point_at], self.y_train[self.knee_point_at], c='b', s=30,
                   label='knee point')
        ax.plot_surface(self.xp, self.yp, self.zp, alpha=0.5)
        plt.legend()
        plt.show()
        

