import numpy as np


class Coordinates(object):
    def __init__(self):
        self.xs = []
        self.ys = []
        self.ws = []
        self.hs = []
        self.max_width = 0
        self.max_height = 0

    def add_coords_from_points(self, points):
        offset = 80
        max_x = np.max(points[:, :, 0])
        min_x = np.min(points[:, :, 0])
        max_y = np.max(points[:, :, 1])
        min_y = np.min(points[:, :, 1])
        mean_x = (max_x + min_x)/2
        mean_y = (max_y + min_y) / 2
        x = mean_x - offset
        y = mean_y - offset
        w = offset * 2
        h = offset * 2
        self.add_hand((x, y, w, h))

    def add_hand(self, hand_cords):
        x, y, w, h = hand_cords
        Coordinates.add_to_collection(self.xs, x)
        Coordinates.add_to_collection(self.ys, y)
        Coordinates.add_to_collection(self.ws, w)
        Coordinates.add_to_collection(self.hs, h)

    @staticmethod
    def add_to_collection(collection, coord):
        if len(collection) > 5:
            collection.pop(0)
        collection.append(coord)

    def has_cords(self):
        return len(self.xs) != 0

    def get_processed_cords(self):
        x, y, w, h = self.get_coords()
        offset = 60
        x -= offset
        y -= offset
        w += 2 * offset
        h += 2 * offset
        x = np.clip(x, 0, self.max_width - 1)
        y = np.clip(y, 0, self.max_height - 1)
        w = np.clip(w + x, 0, self.max_width - 1) - x
        h = np.clip(h + y, 0, self.max_height - 1) - y
        return x, y, w, h

    def get_coords(self):
        return np.mean(self.xs).astype('int'), \
               np.mean(self.ys).astype('int'), \
               np.mean(self.ws).astype('int'), \
               np.mean(self.hs).astype('int'), \


    def clear_all(self):
        self.xs = []
        self.ys = []
        self.ws = []
        self.hs = []
