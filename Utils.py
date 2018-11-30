import numpy as np
import pickle
import cv2 as cv
import os
import matplotlib
import time
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

#calculate the meassures of the arrow
def get_angle_paramerters(last_meassurs, speed, accel, steer_angle):

    arrow_length = speed # length of arrow, detirmined by speed of car
    arrow_width = 0.3# arrow width, detirmined by acceleration
    if steer_angle < 0.4:
        arrow_courve = 0.4
    elif steer_angle >0.7:
        arrow_courve = 0.7
    else:
        arrow_courve = steer_angle # arrow courvage, detirmined by steer_angle

    arrow_color = 'green'

    return arrow_length, arrow_width, arrow_courve, arrow_color

# get the array data
def get_arrow(arrow_length, arrow_width, arrow_courve, arrow_color):
    x_tail = 0.7
    y_tail = 0.16

    x_head = x_tail+arrow_courve-0.5
    y_head = y_tail+arrow_length
    dx = x_head - x_tail
    dy = y_head - y_tail

    fig, axs = plt.subplots()
    arrow = mpatches.Arrow(x_tail, y_tail, dx, dy, width=arrow_width, color=arrow_color)
    axs.add_patch(arrow)
    axs.set_axis_off()
    fig.add_axes(axs)

    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data

def add_arrow_to_frame(frame, arrow):
    # return frame.paste(arrow, (0, 0), arrow)
    return cv.addWeighted(frame[:,:640], 0.6, arrow,0.1,0)


def run():
    with open(r'Data\car_data.pickle', 'rb') as f:
        car_data = pickle.load(f)

    time_stamps = car_data['time_stamp']

    veh_disp_spd = car_data['veh_disp_spd']
    miny = min(veh_disp_spd)
    maxy = 120
    veh_disp_spd_normalized = [(x - miny)/(maxy-miny) for x in veh_disp_spd]

    veh_acc = car_data['veh_acc']
    miny = min(veh_acc)
    maxy = max(veh_acc)
    veh_acc_normalized = [(x - miny)/(maxy-miny) for x in veh_acc]

    stw_ang = car_data['stw_ang']

    miny = -105
    maxy = 105
    stw_ang_normalized = [(x - miny)/(maxy-miny) for x in stw_ang]

    img_paths  = os.

    for ti,spe,accs,steer in zip(time_stamps,veh_disp_spd_normalized,veh_acc_normalized,stw_ang_normalized):
        last_meassurs = None
        print('**** TIMESTEP:',ti,'********')
        print('Speed:', spe, )
        print('Acceleration:', accs, )
        print('Steering Anlg:', steer, )

        # img = cv.imread('Data/imgs/cam_img1540370185_RGB_ALL.jpg',2)
        img = cv.imread('Data/imgs/sample.jpg', 1)
        arrow_length, arrow_width, arrow_courve, arrow_color = get_angle_paramerters(last_meassurs, spe,accs,steer)
        last_meassurs = [spe,accs,steer]
        arrow = get_arrow(arrow_length, arrow_width, arrow_courve, arrow_color)
        frame_with_arrow = add_arrow_to_frame(img, arrow)
        cv.imshow('Car_View', frame_with_arrow)
        cv.waitKey(1)


run()
