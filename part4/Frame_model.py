from part3.SFM_standAlone import FrameContainer

class Frame_model:
    def __init__(self, frame,index):
        self.name = frame
        self.index = index
        self.tfl_lights = []
        self.container = FrameContainer(frame)

    # def add_tfl_light(self,li):