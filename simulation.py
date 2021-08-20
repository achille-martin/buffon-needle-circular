#!/usr/bin/env python

import numpy as np
import math
import random


class Needle:
    def __init__(self,
                 x=0,
                 y=0,
                 length=0,
                 alpha=0
                 ):
        self.cog_coord = np.array([x, y])
        self.length = length
        self.angle = alpha    # rad (from [0 to pi[)

    def get_endpoint_arr(self):
        # Define endpoints as the tips of the needle
        local_endpt_A = np.array([self.length/2 * math.cos(self.angle),
                                  self.length/2 * math.sin(self.angle)
                                  ])
        # Second endpoint is symmetric to the first about the CoG
        local_endpt_B = local_endpt_A * (-1)
        # Create a 2D array with first row corresponding to first endpoint
        # and second row corresponding to second endpoint
        global_endpt_arr = np.array([np.add(self.cog_coord, local_endpt_A),
                                     np.add(self.cog_coord, local_endpt_B)
                                     ])
        return global_endpt_arr


class CircularBoard:
    def __init__(self,
                 min_d=0,
                 max_d=0,
                 step_d=0
                 ):
        self.min_diameter = min_d
        self.max_diameter = max_d
        self.step_diameter = step_d
        self.centre_coord = np.array([0, 0])

    def get_diameter_arr(self):
        # 1D array of diameter values of the concentric circles of the board
        diameter_list = []
        diameter_start = self.min_diameter
        diameter_current = diameter_start
        while diameter_current <= self.max_diameter:
            diameter_list.append(diameter_current)
            diameter_current += self.step_diameter
        diameter_arr = np.array(diameter_list)
        return diameter_arr

    def check_intersection(self, needle=Needle()):
        # Determine whether the needle intersects a circle or not
        # First step is to find the nearest diameter to the CoG of the needle
        diameter_arr = self.get_diameter_arr()
        r_cog_needle = self.get_distance(self.centre_coord, needle.cog_coord)
        nearest_diam_index = (np.abs(diameter_arr - r_cog_needle*2)).argmin()
        nearest_diameter = diameter_arr[nearest_diam_index]

        # Second step is to check whether both endpoints of the needle
        # are on the same side of the diameter
        endpoint_arr = needle.get_endpoint_arr()
        r_endpoint_A = self.get_distance(self.centre_coord, endpoint_arr[0])
        r_endpoint_B = self.get_distance(self.centre_coord, endpoint_arr[1])
        if (r_endpoint_A*2 > nearest_diameter):
            if (r_endpoint_B*2 > nearest_diameter):
                bool_intersect = False
            else:
                bool_intersect = True
        elif (r_endpoint_A*2 < nearest_diameter):
            if (r_endpoint_B*2 < nearest_diameter):
                bool_intersect = False
            else:
                bool_intersect = True
        else:
            bool_intersect = True

        return bool_intersect

    def check_target_bounds(self, point=np.array([0, 0])):
        r_point = self.get_distance(self.centre_coord, point)
        if (r_point*2 > self.max_diameter):
            within_target_bounds = False
        else:
            within_target_bounds = True
        return within_target_bounds

    def get_distance(self, point_A=np.array([0, 0]), point_B=np.array([0, 0])):
        return np.sqrt((point_A[0] - point_B[0])**2 +
                       (point_A[1] - point_B[1])**2
                       )


class Simulator:
    def __init__(self,
                 board_type='circular',
                 nb_board=0,
                 nb_needle_per_board=0,
                 needle_distribution='uniform'
                 ):
        self.board_type = board_type
        self.nb_board = nb_board
        # The board array is a 1D array which contains all board instances
        self.board_arr = np.full(nb_board, None)
        self.nb_needle_per_board = nb_needle_per_board
        # The needle array is a 2D array
        # Each row of the needle array corresponds to a board
        # and contains a 1D array of needles
        self.needle_arr = np.full([nb_board, nb_needle_per_board], None)
        self.needle_distribution = needle_distribution
        # The intersection array is a 2D array describing whether
        # a needle intersects or not
        # Each row corresponds to a board
        # and contains a 1D array of bool corresponding to needle intersections
        self.intersection_arr = np.full([nb_board, nb_needle_per_board],
                                        False,
                                        dtype=bool
                                        )

    def generate_circular_board_arr(self,
                                    min_diameter=0,
                                    max_diameter=0,
                                    step_diameter=0
                                    ):
        board_obj = CircularBoard(min_diameter,
                                  max_diameter,
                                  step_diameter
                                  )
        circular_board_arr = np.full(self.nb_board, board_obj)
        return circular_board_arr

    def generate_needle_on_board_arr(self,
                                     board_type='circular',
                                     needle_distribution='uniform',
                                     nb_needle=0,
                                     needle_length=0,
                                     board_obj=CircularBoard()
                                     ):
        needle_on_board_arr = np.full(nb_needle, None)
        if board_type == 'circular':
            if needle_distribution == 'uniform':
                # Get target boundaries
                diameter_arr = board_obj.get_diameter_arr()
                max_radius = np.amax(diameter_arr)/2
                # Create needles and check they are on target
                for needle_index, needle_obj in enumerate(needle_on_board_arr):
                    while True:
                        r = max_radius * math.sqrt(random.uniform(0, 1))
                        theta = random.uniform(0, 1) * 2 * math.pi
                        alpha = random.uniform(0, 1) * 2 * math.pi
                        new_needle_obj = Needle(
                                board_obj.centre_coord[0] + r*math.cos(theta),
                                board_obj.centre_coord[1] + r*math.sin(theta),
                                needle_length,
                                alpha
                                )
                        needle_endpts = new_needle_obj.get_endpoint_arr()
                        cond1 = board_obj.check_target_bounds(needle_endpts[0])
                        cond2 = board_obj.check_target_bounds(needle_endpts[1])
                        on_target = cond1 and cond2
                        if(on_target):
                            break
                    # Add the new needle to the needle on board arr
                    needle_on_board_arr[needle_index] = new_needle_obj
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return needle_on_board_arr

    def get_intersect_proba_arr(self):
        intersect_proba_arr = np.full(self.nb_board, 0.0)
        for board_index, intersect_arr in enumerate(self.intersection_arr):
            sum_true = intersect_arr.sum()
            intersect_proba = sum_true / float(self.nb_needle_per_board)
            intersect_proba_arr[board_index] = intersect_proba
        return intersect_proba_arr

    def run_simulation(self):
        # Populate the board array depending on board type
        # Populate needle array depending on board type
        if self.board_type == 'circular':
            self.board_arr = self.generate_circular_board_arr(0.05, 0.3, 0.05)
            for board_index, board_obj in enumerate(self.board_arr):
                needle_on_board_arr = self.generate_needle_on_board_arr(
                        self.board_type,
                        self.needle_distribution,
                        self.nb_needle_per_board,
                        0.02,
                        board_obj
                        )
                self.needle_arr[board_index] = needle_on_board_arr
        else:
            raise NotImplementedError
        # Determine the intersections
        for board_index, needle_on_board_arr in enumerate(self.needle_arr):
            for needle_index, needle_obj in enumerate(needle_on_board_arr):
                board_obj = self.board_arr[board_index]
                is_intersect = board_obj.check_intersection(
                        self.needle_arr[board_index, needle_index])
                self.intersection_arr[board_index, needle_index] = is_intersect

    def plot_results(self):
        pass


def main():
    sim_obj = Simulator('circular', 1, 1000000, 'uniform')
    sim_obj.run_simulation()
    intersect_proba_arr = sim_obj.get_intersect_proba_arr()
    print("Intersection probability per board: " + str(intersect_proba_arr))


if __name__ == "__main__":
    main()
