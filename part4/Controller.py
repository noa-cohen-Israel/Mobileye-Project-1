from TFL_Manager import TFL_Manager
from part4.veiw import Veiw


class Controller:
    def __init__(self, playlist):
        with open(playlist, "r") as playlist:
            self.playlist = playlist.readlines()

        self.validate_pls()
        self.tfl_manager = TFL_Manager(self.playlist[0][:-1])  # self.playlist[0] == pkl file
        self.view = Veiw()

    def run(self):
        for i in range(len(self.playlist) - 2):
            tfl_man_part1, tfl_man_part_2_and_3 = self.tfl_manager.run_frame(i + int(self.playlist[1][:-1]),
                                                                             self.playlist[i + 2][:-1])
            if i > 0:
                self.view.veiw_frame(tfl_man_part1, tfl_man_part_2_and_3)

    def validate_pls(self):
        if self.playlist[0][-4:-1] != "pkl":
            raise ValueError("Missing pkl file")

        if not self.playlist[1][:-1].isdigit():
            raise ValueError("Missing number of first file to start from")

# controller = Controller("play_list.pls")
# controller.run()
