import pickle

import numpy as np
from PIL import Image

from Frame_model import Frame_model
from part3 import SFM
from part3.SFM_standAlone import visualize
from part2.part_2 import crop_image_and_verify_tfl
from part1.part1 import find_tfl_lights, show_image_and_gt


class TFL_Manager:
    def __init__(self, pkl_path):
        self.pkl_path = pkl_path
        self.curr_frame = None
        self.prev_frame = None
        self.data = self.load_pickle_data(self.pkl_path)
        self.focal, self.pp = self.data['flx'], self.data['principle_point']

    def run_frame(self, index, frame):
        # initializing frame_model for current frame
        self.curr_frame = Frame_model(frame, index)

        # running part-1 - retrieving tfl candidates
        candidates = self.part1(self.curr_frame.name)
        self.curr_frame.tfl_lights = [[[x, y] for x, y in zip(candidates["red_x"], candidates["red_y"])],
                                      [[x, y] for x, y in zip(candidates["green_x"], candidates["green_y"])]]

        # running part-2 - verifying the candidates above - disqualifies non_tfl candidates and returns true candidates
        self.run_part2()
        self.curr_frame.container.traffic_light = self.curr_frame.tfl_lights
        # running part3 - calculates distance of tfl from cameras position and displays it
        if len(self.curr_frame.tfl_lights[0]) > 0 or len(self.curr_frame.tfl_lights[1]) > 0: # if there are any tfls
            if self.prev_frame:
                self.run_part3()

            # saving current frame for next frame
            tmp_prev=self.prev_frame
            self.prev_frame = self.curr_frame

        return candidates, {"prev_frame":tmp_prev ,"curr_frame":self.curr_frame,"focal":self.focal, "pp":self.pp}

    @staticmethod
    def load_pickle_data(pkl_file):
        """
        loads pickle file data
        :param pkl_file: path to pickle file
        :return: pickle_data
        """
        with open(pkl_file, 'rb') as pklfile:
            data = pickle.load(pklfile, encoding='latin1')
        return data

    def part1(self, image_path):
        red_x, red_y, green_x, green_y = find_tfl_lights(np.array(self.curr_frame.container.img), image_path=image_path,
                                                         some_threshold=42)
        # show_image_and_gt(self.curr_frame.container.img, None, )
        return {"red_x":red_x,"red_y":red_y,"green_x":green_x, "green_y": green_y}

    def run_part2(self):
        for color in range(2):
            candidates_percentage = crop_image_and_verify_tfl(self.curr_frame.name, self.curr_frame.tfl_lights[color])
            verified_tfls = []
            for i in range(len(candidates_percentage)):
                if candidates_percentage[i][1] > 0.55:
                    verified_tfls.append(self.curr_frame.tfl_lights[color][i])
            try:
                assert len(verified_tfls) <= len(self.curr_frame.tfl_lights[color])
            except AssertionError as msg:
                print(msg, ": validated tfls should be less than suspicious ones!")
            self.curr_frame.tfl_lights[color] = np.array(verified_tfls)

    def run_part3(self):
        EM = np.eye(4)
        for i in range(self.prev_frame.index, self.curr_frame.index):
            EM = np.dot(self.data['egomotion_' + str(i) + '-' + str(i + 1)], EM)

        self.curr_frame.container.EM = EM
        self.curr_frame.container = SFM.calc_TFL_dist(self.prev_frame.container,
                                                      self.curr_frame.container, self.focal, self.pp)
