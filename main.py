import sys
sys.path.append("part4")
from part4.Controller import Controller


def run_project():
    controller = Controller("./data/play_list.pls")
    controller.run()


if __name__ == "__main__":
    run_project()