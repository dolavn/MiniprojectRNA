from tkinter import Tk, Canvas, Frame, BOTH
import numpy as np

vector = np.array


class Circle:

    def __init__(self, center, radius, elem_list):
        self.center = center
        self.radius = radius
        self.elem_list = elem_list

    def get_next_ind(self, ind):
        ind_in_list = self.elem_list.index(ind)
        return self.elem_list[(ind_in_list+1) % len(self.elem_list)]

    def __str__(self):
        return "Center:" + str(self.center) + "\nRadius:" + str(
            self.radius) + "\nrange:" + str(self.elem_list)


class Surface(Frame):

    BASE_COLOR = "#888"
    BASE_PAIR_LENGTH = 50
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
        index_list = []
        prev_list = last_circle.elem_list
        last_ind = prev_list.index(last)
        curr = prev_list.index(first)
        while curr != last_ind:
            index_list.append(prev_list[curr])
            curr = (curr+1) % len(prev_list)
        return index_list

    @staticmethod
    def get_angle_with_x(p1: vector, p2: vector) -> float:
        v = p2-p1
        x = np.array([1, 0])
        v = v/np.linalg.norm(v)
        return np.arccos(np.dot(v, x))

    @staticmethod
    def get_base_pair_new_location(p1: vector, p2: vector):
        dist = np.sqrt(np.dot(
            p2 - p1, p2 - p1
        ))
        target_ratio = Surface.BASE_PAIR_LENGTH/dist
        alpha = (1/target_ratio)-1
        beta = alpha+2
        return (p1*beta+p2*alpha)/(alpha+beta), (p1*alpha+p2*beta)/(alpha+beta)

    def get_new_circle(self, angle, first, last, index_list, reverse=False) -> Circle:
        old_center = (
            self.sequence[first]['location'] + self.sequence[last]['location'])/2
        sign = -1 if reverse else 1
        radius = Surface.ALIGNMENT_RADIUS*len(
            index_list) / len(self.sequence)
        new_center = old_center + sign*(radius+30)*np.array(
            [np.cos(angle), np.sin(angle)])
        return Circle(new_center, radius, index_list)

    def init_circle(self, first: int, last: int, reverse: bool=False) -> None:
        last_circle = self.sequence[first]['circle']
        index_list = Surface.get_index_list(first, last, last_circle)
        if len(index_list) == 0:
            return
        quantum = self.ANGLES / len(index_list)
        angle = 0
        if last_circle == 0:  # no last circle
            circle = Circle(self.CENTER, self.ALIGNMENT_RADIUS, index_list)  # first circle
        else:
            ind1 = last if reverse else first-1
            ind2 = first-1 if reverse else last
            angle1 = self.sequence[ind1]['angle']
            angle2 = self.sequence[ind2]['angle']
            angle = np.average([angle1, angle2])
            circle = self.get_new_circle(angle, ind1, ind2, index_list, reverse)
            angle = angle+np.pi if not reverse else angle
            angle = angle + quantum/2

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
        old_circle = self.sequence[ind1]['circle']
        location_new = Surface.get_base_pair_new_location(location1, location2)
        self.sequence[ind1]['location'] = location_new[0]
        self.sequence[ind2]['location'] = location_new[1]
        self.init_circle(old_circle.get_next_ind(ind2), ind1, True)
        self.init_circle(old_circle.get_next_ind(ind1), ind2)
        print(self.sequence[old_circle.get_next_ind(ind2)]['circle'])
        print(self.sequence[old_circle.get_next_ind(ind1)]['circle'])
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
                    other_location = self.sequence[other_ind]['location']
                    location = self.sequence[ind]['location']
                    canvas.create_line(location[0], location[1], other_location[0], other_location[1])

        for ind in range(len(self.sequence)):
            base = self.sequence[ind]
            location = base['location']
            center = base['circle'].center
            #canvas.create_oval(center[0], center[1], center[0]+self.RADIUS, center[1] + self.RADIUS, fill="#11A")
            #canvas.create_oval(location[0], location[1], location[0]+self.RADIUS,
            #                   location[1]+self.RADIUS, fill=self.BASE_COLOR)
            canvas.create_text(location[0]+self.RADIUS/2, location[1]-10, text=base['ind'])
            if ind < len(self.sequence)-1:
                next_base = self.sequence[ind+1]
                location_next = next_base['location']
                canvas.create_line(location[0], location[1], location_next[0], location_next[1])

        canvas.pack(fill=BOTH, expand=1)

root = Tk()
surf = Surface("AGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGU")

surf.add_bp(0, 44)
surf.add_bp(1, 43)
surf.add_bp(2, 42)
surf.add_bp(12, 38)
surf.add_bp(19, 31)
surf.add_bp(20, 30)
#surf.add_bp(17, 25)
#surf.add_bp(6, 9)
surf.init_image()
root.geometry("600x600")
root.mainloop()
