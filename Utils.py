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
def get_angle_paramerters(last_meassurs, speed, accel, steer_angle, no_break):

    arrow_length = speed # length of arrow, detirmined by speed of car
    arrow_width = 0.3# arrow width, detirmined by acceleration
    if steer_angle < 0.35:
        arrow_courve = 0.4
    elif steer_angle >0.7:
        arrow_courve = 0.75
    else:
        arrow_courve = steer_angle # arrow courvage, detirmined by steer_angle

    if no_break == 0:
        arrow_color = 'red'
    # elif last_meassurs is not None and speed+0.01 < last_meassurs[0]:
    #     arrow_color = 'red'
    else:
        arrow_color = 'green'

    return arrow_length, arrow_width, arrow_courve, arrow_color

# get the array data
def get_arrow(arrow_length, arrow_width, arrow_courve, arrow_color):
    # x_tail = 0.5
    # y_tail = 0.16
    #
    # x_head = x_tail+arrow_courve-0.5
    # y_head = y_tail+arrow_length
    # dx = x_head - x_tail
    # dy = y_head - y_tail
    #
    fig, axs = plt.subplots()
    axs.set_xlim(-.5, 0.5)
    axs.set_ylim(0, 1.25)
    # arrow = mpatches.Arrow(x_tail, y_tail, dx, dy, width=arrow_width, color=arrow_color)

    style = ("Simple,tail_width=" + str(50) + ",head_width=" + str(100) + ",head_length=" + str(30))
    kw = dict(arrowstyle=style, color=arrow_color)

    arrow = mpatches.FancyArrowPatch((0, 0), ((arrow_courve - 0.5), 0.3 + arrow_length*1.25), connectionstyle=("arc3,rad=" + str(-1 * (arrow_courve - 0.5))), **kw)

    axs.add_patch(arrow)
    axs.set_axis_off()
    fig.add_axes(axs)

    fig.canvas.draw()
    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return data

def add_arrow_to_frame(frame, arrow):
    arrow_white = np.zeros(frame.shape, dtype='uint8')+255
    x_plus = 100
    y_plus = 220
    arrow_white[x_plus:480+x_plus, y_plus:640+y_plus] = arrow
    arrow_white = cv.cvtColor(arrow_white, cv.COLOR_BGR2RGB)

    frame_with_arrow = cv.addWeighted(frame, 0.8, arrow_white,0.2,0.1)
    return frame_with_arrow


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

    miny = -90
    maxy = 90
    stw_ang_normalized = [(x - miny)/(maxy-miny) for x in stw_ang]

    base_dir = 'Data/img_out'
    img_paths = os.listdir(base_dir)

    with open(r'Data\inmylane.pickle', 'rb') as f:
        in_my_lane = pickle.load(f)
    in_my_lane = [elem[1] for elem in in_my_lane]

    lane_no = 0.0
    im_no = 0.0
    last_meassurs = None
    for ti,spe,accs,steer in zip(time_stamps,veh_disp_spd_normalized,veh_acc_normalized,stw_ang_normalized):

        print('**** TIMESTEP:',ti,'********')
        print('Speed:', spe, )
        print('Acceleration:', accs, )
        print('Steering Anlg:', steer, )

        im_no_int = int(round(im_no))
        lane_no_int = int(round(lane_no))
        img = cv.imread(os.path.join(base_dir,img_paths[im_no_int]), 1)
        no_break = in_my_lane[lane_no_int]
        im_no += 1.9
        lane_no += 1
        arrow_length, arrow_width, arrow_courve, arrow_color = get_angle_paramerters(last_meassurs, spe,accs,steer, no_break)
        last_meassurs = [spe,accs,steer]
        arrow = get_arrow(arrow_length, arrow_width, arrow_courve, arrow_color)
        frame_with_arrow = add_arrow_to_frame(img, arrow)
        cv.imshow('Car_View', frame_with_arrow)
        cv.waitKey(1)


run()
