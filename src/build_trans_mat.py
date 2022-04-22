import os
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def build_matrix(H, Zs, train, test, n_stop, alpha=1.0):
    F = np.zeros((n_stop, n_stop))
    print(H.shape)

    for k in range(H.shape[0]):
        Hk = H[k, :, :]
        # Sigma = np.where(Hk.sum(axis=1) > 0)[0]
        # print(k, Sigma)

        Ak = np.zeros_like(F)
        Ak[np.where(Hk > 0)] = 1
        F += Ak

    T = np.zeros_like(F)
    Trow = T.sum(axis=1)
    for (i, j) in product(range(n_stop), range(n_stop)):
        T[i, j] = (F[i, j] + alpha) / (Trow[i] + alpha * n_stop)
    return T


def read_data(path="./data"):
    path_route = os.path.join(path, "daily_routematrix.npz")
    Zr = np.load(path_route, allow_pickle=True)
    keys_Zr = list(Zr.keys())
    print(f"Keys: {keys_Zr}")

    n_stop = len(Zr["stop_wise_active"])
    print(n_stop)

    path_stop = os.path.join(path, "daily_stops.npz")
    Zs = np.load(path_stop, allow_pickle=True)
    weekday = Zs["weekday"]
    list_days = sorted(set(weekday))
    keys_Zs = list(Zs.keys())
    print(f"Keys: {keys_Zs}")

    # WD -> data index
    dict_wd_index = {day: np.where(weekday == day)[0] for day in list_days}
    print(dict_wd_index[0])

    for target in list_days:
        train, test = train_test_split(dict_wd_index[target], test_size=0.25)
        train.sort()
        test.sort()
        H = Zr["incidence_matrices"][train]
        print(f" Train: {train}")
        print(f" Test : {test}")
        T = build_matrix(H, Zs, train, test, n_stop)

        # visualize
        f = plt.figure()
        a = f.gca()
        a.imshow(T, cmap="jet", aspect=1)
        a.set_xlabel("Index of stops")
        a.set_ylabel("Index of stops")
        plt.tight_layout()
        plt.savefig(f"figures/example_trans_mat_{target}.png")
        plt.close()


def build_trans_mat():
    raise NotImplementedError()


if __name__ == "__main__":
    read_data()
    # build_trans_mat()
