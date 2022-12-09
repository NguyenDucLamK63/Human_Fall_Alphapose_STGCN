
# import os
 
# # Get the list of all files and directories
# path = "/home/duclam/Documents/dataset_action/Le2i_FDD_fall/Home/Videos"
# dir_list = sorted(os.listdir(path))
 
# print("Files and directories in '", path, "' :")
 
# prints all files
# print(dir_list)
import cv2
import numpy as np
import glob
import argparse
import os
def parse_args():
    parser = argparse.ArgumentParser(description='crop_frame demo')
    #1080x1920x3 size : 6220800
    parser.add_argument(
        '--video',
        default='/home/duclam/Lam/fall_detection/Human-Falling-Detect-Tracks_2/test_cty_23_11_2person.mp4',
        help='video file/url')
    parser.add_argument(
    '--folder_video',
    default='/home/duclam/Documents/dataset_action/Home_test/video_test/Videos',
    help='video folder')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--output',
        default='/home/duclam/Lam/fall_detection/Human-Falling-Detect-Tracks_2/Data/ouput/cafe',
        help='output filename')
    parser.add_argument(
        '--output-fps',
        default=24,
        type=int,
        help='the fps of demo video output')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = parse_args()
    img_array = []
    i = 1
    new_name = 'video_{:02d}.avi'
    # print(new_name.format(i))
    # for filename in sorted(glob.glob(args.folder_video + '/*.avi')):
    #     # img = cv2.imread(filename)
    #     # height, width, layers = img.shape
    #     # size = (width,height)
    #     print(filename)
    #     new_file = os.rename(filename, new_name.format(i))
    #     i = i + 1
    #     # img_array.append(img)
    #     # print(size)
    #     # print(img_array)
    # folder = "xyz"
    # for count, filename in sorted(enumerate(os.listdir(args.folder_video))):
    #     count = count + 1
        # dst = f"video_{str(count)}.avi"
        
        # src =f"{args.folder_video}/{filename}"  # foldername/filename, if .py file is outside folder
        # dst =f"{args.folder_video}/{dst}"
    count = 1
    for filename in sorted(os.listdir((args.folder_video))):
        print(filename) 
        dst = f"video_{str(count)}.avi"
        
        src =f"{args.folder_video}/{filename}"  # foldername/filename, if .py file is outside folder
        dst =f"{args.folder_video}/{dst}"
        os.rename(src, dst)
        print(count)
        count += 1
        
        # rename() function will
        # rename all the files
       

# out = cv2.VideoWriter('/home/duclam/Documents/dataset_action/UP-Fall_detection_dataset/Subject1Activity7Trial1Camera1.avi',cv2.VideoWriter_fourcc(*'DIVX'), 25, size)
# # out.write(img_array)
# print(len(img_array))
# for i in range(len(img_array)):
#     out.write(img_array[i])
# out.release()