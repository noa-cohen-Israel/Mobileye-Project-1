import matplotlib.pyplot as plt

from part1.part1 import show_image_and_gt
from part3.SFM_standAlone import visualize


class Veiw:
    def __init__(self):
        pass

    def veiw_frame(self, part_1, part_2_and_3):
        fig = plt.figure(figsize=(8, 12))
        fig.add_subplot(3, 1, 1)

        show_image_and_gt(part_2_and_3["curr_frame"].container.img, None, part_1["red_x"], part_1["red_y"], part_1["green_x"],
                          part_1["green_y"])
        plt.axis('off')
        plt.title("light sources") #part1
        fig.add_subplot(3, 1, 2)

        show_image_and_gt(part_2_and_3["curr_frame"].container.img, None, part_2_and_3["curr_frame"].tfl_lights[0][:, 0],
                          part_2_and_3["curr_frame"].tfl_lights[0][:, 1], part_2_and_3["curr_frame"].tfl_lights[1][:, 0],
                          part_2_and_3["curr_frame"].tfl_lights[1][:, 1])
        plt.axis('off')
        plt.title("TFLs")#part2
        fig.add_subplot(3, 1, 3)
        visualize(part_2_and_3["prev_frame"].container, part_2_and_3["prev_frame"].index, part_2_and_3["curr_frame"].container,
                  part_2_and_3["curr_frame"].index, part_2_and_3["focal"], part_2_and_3["pp"])
        plt.axis('off')
        plt.title("Distances")#part3

        plt.text(8, 1200,"By: Noa C. & Chava N.")
        plt.show()
