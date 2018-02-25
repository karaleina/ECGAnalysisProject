import numpy as np

def calculate_area_under_curve(x=[], y=[]):

    prev_el_x = None
    prev_el_y = None
    current_area = 0

    for (el_x, el_y) in zip(x,y):

        if prev_el_x is not None and prev_el_y is not None:
            current_area += float((np.abs(el_x - prev_el_x)))*np.abs((el_y + prev_el_y)) / 2

        prev_el_x = el_x
        prev_el_y = el_y

    return current_area


if __name__ == "__main__":

    x = [0,3.5,7]
    y = [4,5,4]

    area = calculate_area_under_curve(x=x, y=y)
    print(area)