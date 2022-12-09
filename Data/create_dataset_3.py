"""
This script to create dataset and labels by clean off some NaN, do a normalization,
label smoothing and label weights by scores.

"""
import os
import pickle
import numpy as np
import pandas as pd


class_names = ['Standing', 'Walking', 'Sitting', 'Lying Down',
               'Stand up', 'Sit down', 'Fall Down', ]
main_parts = ['LShoulder_x', 'LShoulder_y', 'RShoulder_x', 'RShoulder_y', 'LHip_x', 'LHip_y',
              'RHip_x', 'RHip_y']
main_idx_parts = [1, 2, 7, 8, -1]  # 1.5

csv_pose_file = '/home/duclam/Documents/dataset_action/Le2i_FDD_fall/Home/Home_FDD_fix_20_fall_pose_101.csv'
save_path = '/home/duclam/Documents/dataset_action/Le2i_FDD_fall/Home/Home_FDD_fix_20_fall_101.pkl'

# Params.
smooth_labels_step = 8
n_frames = 30
skip_frame = 1

annot = pd.read_csv(csv_pose_file)

# Remove NaN.
idx = annot.iloc[:, 2:-1][main_parts].isna().sum(1) > 0
idx = np.where(idx)[0]
annot = annot.drop(idx)
# One-Hot Labels.
label_onehot = pd.get_dummies(annot['label']) # Chuyen doi du lieu
annot = annot.drop('label', axis=1).join(label_onehot)
cols = label_onehot.columns.values #so label, gan nhan label
# dua ve dang one-hot (0,1,0,0,0) , ....


def scale_pose(xy):
    """
    Normalize pose points by scale with max/min value of each pose. : Max/min
    xy : (frames, parts, xy) or (parts, xy)
    """
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)
    xy_min = np.nanmin(xy, axis=1) #0.64,0.37
    xy_max = np.nanmax(xy, axis=1) #0.7966,
    for i in range(xy.shape[0]):
        xy[i] = ((xy[i] - xy_min[i]) / (xy_max[i] - xy_min[i])) * 2 - 1
    return xy.squeeze()


def seq_label_smoothing(labels, max_step=10):
    steps = 0
    remain_step = 0
    target_label = 0
    active_label = 0
    start_change = 0
    max_val = np.max(labels) #0.9
    min_val = np.min(labels) #0.025
    # duyet i, kiem hanh dong khac
    #53 
    for i in range(labels.shape[0]): #label tu 1 - het video (264)
        if remain_step > 0:
            if i >= start_change:
                labels[i][active_label] = max_val * remain_step / steps
                labels[i][target_label] = max_val * (steps - remain_step) / steps \
                    if max_val * (steps - remain_step) / steps else min_val
                remain_step -= 1
            continue

        diff_index = np.where(np.argmax(labels[i:i+max_step], axis=1) - np.argmax(labels[i]) != 0)[0] #array([7])
        # tao mang, kiem hanh dong khac voi hanh dong truoc, gan vao index
        if len(diff_index) > 0:
            start_change = i + remain_step // 2 #remain_step = 7 (or = 0?) , start = 53
            steps = diff_index[0]
            remain_step = steps
            target_label = np.argmax(labels[i + remain_step])
            active_label = np.argmax(labels[i])
    return labels


feature_set = np.empty((0, n_frames, 14, 3)) #mang co shape = (0,30,14,3)
labels_set = np.empty((0, len(cols))) #shape = (0,5)
vid_list = annot['video'].unique() # list so video
for vid in vid_list:
    print(f'Process on: {vid}')
    data = annot[annot['video'] == vid].reset_index(drop=True).drop(columns='video') #xoa column = video

    # Label Smoothing.
    esp = 0.1 # data[cols] = so label : 1234567
    # cong thuc tinh label
    data[cols] = data[cols] * (1 - esp) + (1 - data[cols]) * esp / (len(cols) - 1) # dua ra label vao mang , tinh xac xuat label
    # data[cols].values : chuan hoa tat ca data trong video
    data[cols] = seq_label_smoothing(data[cols].values, smooth_labels_step) #buoc lam min label

    # Separate continuous frames. 
    frames = data['frame'].values
    frames_set = []
    fs = [0]
    for i in range(1, len(frames)):
        if frames[i] < frames[i-1] + 10: #2<11 fs = 264
            fs.append(i)
        else:
            frames_set.append(fs)
            fs = [i]
    frames_set.append(fs)

    for fs in frames_set:
        xys = data.iloc[fs, 1:-len(cols)].values.reshape(-1, 13, 3) #luu keypoint 13 diem (264,13,3)
        # Scale pose normalize.
        xys[:, :, :2] = scale_pose(xys[:, :, :2]) #khong lay nose_s , float 64 #xuly
        # Add center point.
        xys = np.concatenate((xys, np.expand_dims((xys[:, 1, :] + xys[:, 2, :]) / 2, 1)), axis=1) #them 1 point
        #Shape : 264:14:3
        # Weighting main parts score.
        # Cân điểm các bộ phận chính.
        scr = xys[:, :, -1].copy() #laydiemcuoi 
        # dua diem score : 1,2,7,8,13 ve 1
        scr[:, main_idx_parts] = np.minimum(scr[:, main_idx_parts] * 1.5, 1.0) #(264,5)  #???
        # Mean score.
        scr = scr.mean(1) # lay trung binh

        # Targets.
        lb = data.iloc[fs, -len(cols):].values #lay lai xac xuat raw car cac class sap chuyen (6)
        # Apply points score mean to all labels.
        lb = lb * scr[:, None] #xacxuat * point score trung binh

        for i in range(xys.shape[0] - n_frames):
            feature_set = np.append(feature_set, xys[i:i+n_frames][None, ...], axis=0)
            labels_set = np.append(labels_set, lb[i:i+n_frames].mean(0)[None, ...], axis=0) # trung binhf, sau 15


with open(save_path, 'wb') as f:
    pickle.dump((feature_set, labels_set), f)
