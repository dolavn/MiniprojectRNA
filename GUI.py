from tkinter import Tk, Canvas, Frame, BOTH
from typing import List
from typing import Tuple
import numpy as np
import abc
vector = np.array


class Linkage:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def next(self) -> List['Linkage']:
        return []

    @staticmethod
    def dfs_visit(curr_node: 'Linkage', nodes_list: List['Linkage']):
        if curr_node not in nodes_list:
            nodes_list.append(curr_node)
            print("Visiting " + str(curr_node))
            next_nodes = curr_node.next()
            for next_node in next_nodes:
                Linkage.dfs_visit(next_node, nodes_list)

    @staticmethod
    def run_dfs(first_node: 'Linkage', ignore_node: 'Linkage'):
        nodes_list = [ignore_node]
        Linkage.dfs_visit(first_node, nodes_list)

class BasePair(Linkage):

    def __init__(self, ind1, ind2):
        self.ind1 = ind1
        self.ind2 = ind2
        self.links = []

    def set_links(self, link1: Linkage, link2: Linkage) -> None:
        self.links = [link1, link2]

    def replace_circle(self, new_circle: 'Circle') -> None:
        for i in range(2):
            if isinstance(self.links[i], Circle):
                if new_circle.contained_in(self.links[i]):
                    self.links[i] = new_circle
                    return
        raise Exception("Couldn't replace")

    def set_next_base_pair(self, bp: 'BasePair') -> None:
        for i in range(2):
            if isinstance(self.links[i], Circle):
                if self.links[i].in_circle(bp.ind1):
                    self.links[i] = bp
                    return
            else:  # not a circle, replace other
                self.links[(i+1) % 2] = bp
                return
        raise Exception("Couldn't replace")

    def next(self) -> List[Linkage]:
        """if caller not in self.links:
            for link in self.links:
                print("link:" + str(link))
            print("caller:" + str(caller))
            raise Exception("Caller not linked by this base pair")
        ans = list(self.links)
        ans.remove(caller)"""
        ans = list(self.links)
        return ans

    def __str__(self):
        return "(" + str(self.ind1) + "," + str(self.ind2) + ")"

    def __eq__(self, other):
        if not isinstance(other, BasePair):
            return False
        return self.ind1 == other.ind1 and self.ind2 == other.ind2


class Circle(Linkage):

    def __init__(self, center: vector, radius: float, elem_list: list):
        self.center = center
        self.radius = radius
        self.elem_list = elem_list
        self.bp_list = []

    def next(self) -> List[Linkage]:
        ans = [pair[0] for pair in self.bp_list]
        """if caller is EMPTY_CIRCLE:
            return ans
        if caller not in ans:
            raise Exception("Caller not linked to this circle")
        print(caller)
        ans.remove(caller)"""
        return ans

    def add_bp(self, bp: BasePair, next_bases: Tuple[int, int]) -> None:
        self.bp_list.append((bp, next_bases))

    def in_circle(self, ind: int) -> bool:
        return ind in self.elem_list

    def contained_in(self, other_circle: 'Circle') -> bool:
        max_ind = max(max(self.elem_list), max(other_circle.elem_list))
        binary_arr = np.zeros(max_ind+1, dtype=int)
        for ind in other_circle.elem_list:
            binary_arr[ind] = 1
        for ind in self.elem_list:
            if binary_arr[ind] == 0:
                return False
        return True

    def get_next_ind(self, ind: int) -> int:
        ind_in_list = self.elem_list.index(ind)
        return self.elem_list[(ind_in_list+1) % len(self.elem_list)]

    def get_prev_ind(self, ind: int) -> int:
        ind_in_list = self.elem_list.index(ind)
        return self.elem_list[(ind_in_list-1) % len(self.elem_list)]

    def __str__(self) -> str:
        #return "Center:" + str(self.center) + "\nRadius:" + str(
         #   self.radius) + "\nrange:" + str(self.elem_list)
        return "circle:" + str(self.elem_list)

EMPTY_CIRCLE = Circle(np.array([0, 0]), 0, [])


class Surface(Frame):

    BASE_COLOR = "#888"
    BASE_PAIR_LENGTH = 35
    BASES_DISTANCE = 25
    RADIUS = 20
    ALIGNMENT_RADIUS = 200
    CENTER = np.array([300, 300])
    INIT_X = 10
    ANGLES = 2*np.pi
    INIT_Y = 200

    def __init__(self, sequence):
        super().__init__()
        self.sequence = [{'ind': i, 'base': sequence[i], 'location': np.array([-1, -1]), 'angle':0, 'link':0}
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

    def get_base_pair(self, ind1: int, ind2: int) -> BasePair:
        for bp in self.bp_list:
            if (ind1 == bp.ind1 and ind2 == bp.ind2) or (
                    ind1 == bp.ind2 and ind2 == bp.ind1):
                return bp
        raise Exception("Base pair not found")

    def get_new_circle(self, angle, first, last, index_list, reverse=False) -> Circle:
        old_center = (
            self.sequence[first]['location'] + self.sequence[last]['location'])/2
        sign = -1 if reverse else 1
        #radius = Surface.ALIGNMENT_RADIUS*len(
        #   index_list) / len(self.sequence)
        radius = len(index_list)*self.BASES_DISTANCE/(2*np.pi)
        new_center = old_center + sign*(radius+30)*np.array(
            [np.cos(angle), np.sin(angle)])
        return Circle(new_center, radius, index_list)

    def set_locations_circle(self, circle: Circle, init_angle: float, quantum: float) -> None:
        center = circle.center
        radius = circle.radius
        for i in range(len(circle.elem_list)):
            curr_angle = init_angle + i*quantum
            ind = circle.elem_list[i]
            delta = radius * np.array([np.cos(curr_angle), np.sin(curr_angle)])
            self.sequence[ind]['location'] = center+delta
            self.sequence[ind]['angle'] = curr_angle
            self.sequence[ind]['link'] = circle

    def init_circle(self, first: int, last: int, reverse: bool=False) -> Circle:
        last_circle = self.sequence[first]['link']
        index_list = Surface.get_index_list(first, last, last_circle)
        if len(index_list) == 0:
            return EMPTY_CIRCLE
        quantum = self.ANGLES / len(index_list)
        angle = 0
        if last_circle == 0:  # no last circle
            radius = len(self.sequence)*self.BASES_DISTANCE/(2*np.pi)
            circle = Circle(self.CENTER, radius, index_list)  # first circle
        else:
            ind1 = last if reverse else last_circle.get_prev_ind(first)
            ind2 = last_circle.get_prev_ind(first) if reverse else last
            angle1 = self.sequence[ind1]['angle']
            angle2 = self.sequence[ind2]['angle']
            angle = np.average([angle1, angle2])
            circle = self.get_new_circle(angle, ind1, ind2, index_list, reverse)
            angle = angle+np.pi if not reverse else angle
            angle = angle + quantum/2
        self.set_locations_circle(circle, angle, quantum)
        return circle

    def add_bp(self, ind1: int, ind2: int) -> None:
        if not 0 <= ind1 < len(self.sequence) or not 0 <= ind2 < len(self.sequence):
            raise IndexError("Index out of range")
        if BasePair(ind1, ind2) not in self.bp_list:
            self.bp_list.append(BasePair(ind1, ind2))

    def update_links(self, bp: BasePair, circle1: Circle, circle2: Circle, old_circle: Circle) -> None:
        link1 = circle1
        link2 = circle2
        if circle1 is EMPTY_CIRCLE:
            link1 = self.get_base_pair(bp.ind2+1, bp.ind1-1)
            link1.set_next_base_pair(bp)
        else:
            pair = (old_circle.get_next_ind(bp.ind2), old_circle.get_prev_ind(bp.ind1))
            circle1.add_bp(bp, pair)
        if circle2 is EMPTY_CIRCLE:
            link2 = self.get_base_pair(bp.ind2 + 1, bp.ind1 - 1)
            link2.set_next_base_pair(bp)
        else:
            pair = (old_circle.get_next_ind(bp.ind1), old_circle.get_prev_ind(bp.ind2))
            circle2.add_bp(bp, pair)
        bp.set_links(link1, link2)
        for old_link in old_circle.bp_list:
            pair = old_link[1]
            for link in [circle1, circle2]:
                if link.in_circle(pair[0]) and link.in_circle(pair[1]):
                    link.add_bp(old_link[0], old_link[1])
                    old_link[0].replace_circle(link)

    def create_pair(self, bp: BasePair) -> None:
        location1 = self.sequence[bp.ind1]['location']
        location2 = self.sequence[bp.ind2]['location']
        old_circle = self.sequence[bp.ind1]['link']
        location_new = Surface.get_base_pair_new_location(location1, location2)
        self.sequence[bp.ind1]['location'] = location_new[0]
        self.sequence[bp.ind2]['location'] = location_new[1]
        link1 = self.init_circle(old_circle.get_next_ind(bp.ind2), bp.ind1, True)
        link2 = self.init_circle(old_circle.get_next_ind(bp.ind1), bp.ind2)
        self.update_links(bp, link1, link2, old_circle)

    def init_image(self) -> None:
        self.master.title("RNA Folding")
        self.pack(fill=BOTH, expand=1)
        canvas = Canvas(self)
        for base in self.sequence:
            ind = base['ind']
            for bp in self.bp_list:
                if ind == bp.ind1:
                    self.create_pair(bp)
                    other_location = self.sequence[bp.ind2]['location']
                    location = self.sequence[ind]['location']
                    canvas.create_line(location[0], location[1], other_location[0], other_location[1])
        bpp = self.get_base_pair(32, 38)
        for link in bpp.links:
            print("l" + str(link))
        circ = self.sequence[20]['link']
        #Linkage.run_dfs(circ)
        """
        last = EMPTY_CIRCLE
        curr = self.sequence[16]['link']
        cont = True
        while cont:
            print(curr)
            cont = False
            for next_link in curr.next(last):
                nxt = next_link.next(curr)
                if nxt == []:
                    continue
                else:
                    cont = True
                    last = curr
                    curr = nxt[0]
                    break
        """
        for ind in range(len(self.sequence)):
            base = self.sequence[ind]
            location = base['location']
            canvas.create_text(location[0]+self.RADIUS/2, location[1]-10, text=base['ind'])
            if ind < len(self.sequence)-1:
                next_base = self.sequence[ind+1]
                location_next = next_base['location']
                canvas.create_line(location[0], location[1], location_next[0], location_next[1])

        canvas.pack(fill=BOTH, expand=1)

root = Tk()
surf = Surface("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")


surf.add_bp(2, 10)
surf.add_bp(3, 9)
surf.add_bp(4, 8)
surf.add_bp(5, 7)

surf.add_bp(30, 40)
surf.add_bp(31, 39)
surf.add_bp(32, 38)
surf.add_bp(33, 37)
surf.add_bp(34, 36)


#surf.add_bp(18, 24)
surf.add_bp(19, 23)

#surf.add_bp(7, 10)
#surf.add_bp(17, 25)
#surf.add_bp(6, 9)

surf.init_image()
root.geometry("600x600")
root.mainloop()
