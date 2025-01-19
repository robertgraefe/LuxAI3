import numpy
import matplotlib.pyplot as plt
import matplotlib as mpl



steps = numpy.arange(stop = 1000)
actions = ["north", "east", "south", "west"]
home = [numpy.random.choice(numpy.arange(start=-50, stop=50, step=1)), numpy.random.choice(numpy.arange(start=-50, stop=50, step=1))]

def move(action: str, position: list[int, int]) -> list[int, int]:
    if action == "north":
        position[1] += 1
    elif action == "east":
        position[0] += 1
    elif action == "south":
        position[1] -= 1
    elif action == "west":
        position[0] -= 1
    else:
        raise Exception(f"Action {action} unknown!")
    return position

def walk(action_generator):
    def walking():
        position = [0,0]
        path = [[0,0]]
        for step in steps:
            action = action_generator()
            position = move(action, position)
            path.append(position.copy())
        return path
    return walking

def color_fader(color1: str, color2: str, length: int) -> list:
    color1 = numpy.array(mpl.colors.to_rgb(color1))
    color2 = numpy.array(mpl.colors.to_rgb(color2))
    color_list = list()
    for i in range(length):
        mix = i / length
        new_color = mpl.colors.to_hex((1-mix)*color1 + mix*color2)
        color_list.append(new_color)
    return color_list

def display_path(path) -> None:
    x = [x[0] for x in path]
    y = [x[1] for x in path]
    fig, ax = plt.subplots()
    colors = color_fader(color1="green", color2="#1f77b4", length=len(steps)+1)
    ax.grid(which="both", zorder=0)
    ax.plot(x, y, zorder=1)
    ax.scatter(x, y, c=colors, zorder=2)
    ax.scatter(x=0, y=0, c="yellow", label="Startpunkt", zorder=3)
    ax.scatter(x=x[-1], y=y[-1], c="red", label="Endpunkt", zorder=3)
    ax.scatter(x=home[0], y=home[1], c="black", label="Zuhause", zorder=4)
    ax.legend()
    plt.show()

@walk
def intention_random():
    action = numpy.random.choice(a=actions, size=1, replace=True)
    return action

@walk
def intention_drift_east():
    action = numpy.random.choice(a=actions, size=1, replace=True, p=[.2, .4, .2, .2])
    return action

@walk
def intention_drift_north():
    action = numpy.random.choice(a=actions, size=1, replace=True, p=[.4, .2, .2, .2])
    return action

# display_path(intention_random())
# display_path(intention_drift_east())
# display_path(intention_drift_north())



