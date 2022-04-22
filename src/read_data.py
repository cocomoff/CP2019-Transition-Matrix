import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    path = "./data"
    path_stop = os.path.join(path, "daily_stops.npz")
    Zs = np.load(path_stop, allow_pickle=True)
    keys_Zs = list(Zs.keys())
    print(f"Keys: {keys_Zs}")

    path_dist = os.path.join(path, "Distancematrix.npy")
    dmat = np.load(path_dist)
    nstop = dmat.shape[0]

    stops_list = Zs["stops_list"]
    weekday = Zs["weekday"]
    list_days = sorted(set(weekday))
    used_counter_per_day = np.zeros((7, nstop))
    for day in list_days:
        indices = np.where(weekday == day)[0]
        stops_day = stops_list[indices]
        count_stops_day = list(map(len, stops_day))
        # print(day, count_stops_day)
        for l in stops_day:
            used_counter_per_day[day, l] += 1

    yg = [0, 1, 2, 3, 4, 5, 6]

    f = plt.figure()
    a = f.gca()
    a.imshow(used_counter_per_day, cmap="jet", aspect=3)
    a.set_xlabel("Index of stops")
    a.set_ylabel("WD")
    a.set_yticks(yg)
    a.set_yticklabels(map(str, yg))
    plt.tight_layout()
    plt.savefig("figures/used_stop_per_day.png")
    plt.close()


if __name__ == '__main__':
    main()
