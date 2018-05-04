from tkinter import Tk, Canvas, Frame, BOTH
import numpy as np
from itertools import chain


class Circle:

    def __init__(self, center, radius, first, last):
        self.center = center
        self.radius = radius
        self.first = first
        self.last = last

    def __str__(self):
        return "Center:" + str(self.center) + "\nRadius:" + str(self.radius) + "\n[" + str(
            self.first) + "," + str(self.last) + "]"

vector = np.array

class Surface(Frame):

    BASE_COLOR = "#888"
    RADIUS = 20
    ALIGNMENT_RADIUS = 200
    CENTER = np.array([300, 300])
    INIT_X = 10
    ANGLES = 2*np.pi
    INIT_Y = 200

    def __init__(self, sequence):
        super().__init__()
        self.sequence = [{'ind': i, 'base': sequence[i], 'location': np.array([-1, -1]), 'angle':0, 'circle':0}
                         for i in range(len(sequence))]
        self.bp_list = []
        self.init_circle(0, len(self.sequence))

    @staticmethod
    def get_index_list(first: int, last: int, last_circle: Circle) -> list:
        if last > first:
            return [i for i in range(first, last)]
        prev_first = last_circle.first
        prev_last = last_circle.last
        print(last_circle)
        index_list = [i for i in chain(
            range(first, prev_last), range(prev_first, last))]
        return index_list

    @staticmethod
    def get_angle_with_x(p1: vector, p2: vector) -> float:
        v = p2-p1
        x = np.array([1, 0])
        v = v/np.linalg.norm(v)
        return np.arccos(np.dot(v, x))

    def get_new_circle(self, angle, first, last, index_list, reverse=False) -> Circle:
        old_center = (
            self.sequence[first]['location'] + self.sequence[last]['location'])/2
        sign = -1 if reverse else 1
        radius = Surface.ALIGNMENT_RADIUS*len(
            index_list) / len(self.sequence)
        new_center = old_center + sign*(radius+30)*np.array(
            [np.cos(angle), np.sin(angle)])
        return Circle(new_center, radius, first, last)

    def init_circle(self, first: int, last: int, reverse: bool=False) -> None:
        last_circle = self.sequence[first]['circle']
        index_list = Surface.get_index_list(first, last, last_circle)
        print(index_list)
        angle = 0
        if last_circle == 0:  # no last circle
            circle = Circle(self.CENTER, self.ALIGNMENT_RADIUS, 0, len(self.sequence))  # first circle
        else:
            ind1 = first-1 if not reverse else last
            ind2 = last if not reverse else first-1
            angle1 = self.sequence[ind1]['angle']
            angle2 = self.sequence[ind2]['angle']
            angle = np.average([angle1, angle2])
            circle = self.get_new_circle(angle, ind1, ind2, index_list, reverse)
            loc1 = self.sequence[ind2 if reverse else ind1]['location']
            angle = (-1 if reverse else 1)*Surface.get_angle_with_x(
                circle.center, loc1)

        quantum = self.ANGLES/len(index_list)
        center = circle.center
        radius = circle.radius
        for i in range(len(index_list)):
            curr_angle = angle + i*quantum
            ind = index_list[i]
            delta = radius * np.array([np.cos(curr_angle), np.sin(curr_angle)])
            self.sequence[ind]['location'] = center+delta
            self.sequence[ind]['angle'] = curr_angle
            self.sequence[ind]['circle'] = circle

    def add_bp(self, ind1: int, ind2: int) -> None:
        if not 0 <= ind1 < len(self.sequence) or not 0 <= ind2 < len(self.sequence):
            raise IndexError("Index out of range")
        if not (ind1, ind2) in self.bp_list:
            self.bp_list.append((ind1, ind2))

    def set_locations_loop(self, ind1: int, ind2: int) -> None:
        location1 = self.sequence[ind1]['location']
        location2 = self.sequence[ind2]['location']
        location1_new = (location1*11+location2*9)/20
        location2_new = (location1*9+location2*11)/20
        self.sequence[ind1]['location'] = location1_new
        self.sequence[ind2]['location'] = location2_new
        self.init_circle(ind2+1, ind1, True)
        self.init_circle(ind1+1, ind2)
        # print("ind1:" + str(ind1) + "\nind2:" + str(ind2) + "\nlocation_init:" + str(location_init) + "\nlocation_fin:" + str(location_fin))

    def init_image(self) -> None:
        self.master.title("RNA Folding")
        self.pack(fill=BOTH, expand=1)
        canvas = Canvas(self)
        for base in self.sequence:
            ind = base['ind']
            for other_ind in range(ind+1, len(self.sequence)):
                if (ind, other_ind) in self.bp_list:
                    self.set_locations_loop(ind, other_ind)
                    location1 = self.sequence[ind]['location']
                    location2 = self.sequence[other_ind]['location']
                    location_avg = (location1+location2)/2
                    #canvas.create_oval(
                    #  location_avg[0], location_avg[1], location_avg[0]+self.RADIUS, location_avg[1] + self.RADIUS, fill="#800")

        for base in self.sequence:
            location = base['location']
            center = base['circle'].center
            canvas.create_oval(center[0], center[1], center[0]+self.RADIUS, center[1] + self.RADIUS, fill="#11A")
            canvas.create_oval(location[0], location[1], location[0]+self.RADIUS,
                               location[1]+self.RADIUS, fill=self.BASE_COLOR)
            canvas.create_text(location[0]+self.RADIUS/2, location[1]-10, text=base['ind'])
        canvas.pack(fill=BOTH, expand=1)

root = Tk()
surf = Surface("UAUUAACAAGAAAUAA")
surf.add_bp(2, 12)
#surf.add_bp(3, 11)
#surf.add_bp(1, 3)
#surf.add_bp(6, 9)
surf.init_image()
root.geometry("600x600")
root.mainloop()
