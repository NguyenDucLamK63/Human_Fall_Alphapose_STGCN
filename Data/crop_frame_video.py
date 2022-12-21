import cv2
import argparse
import os.path as osp
import os
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='crop_frame demo')
    #1080x1920x3 size : 6220800
    parser.add_argument(
        '--video',
        default='/home/duclam/Documents/dataset_action/UR_Fall_dataset/Fall/test_fall_cam0/fall-01-cam0-rgb.avi',
        help='video file/url')
    parser.add_argument(
    '--folder_video',
    default='/home/duclam/Documents/dataset_action/UR_Fall_dataset/Fall/test_fall_cam0',
    help='video folder')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--output',
        default='/home/duclam/Lam/fall_detection/Human-Falling-Detect-Tracks_2/Data/ouput',
        help='output filename')
    parser.add_argument(
        '--output-fps',
        default=24,
        type=int,
        help='the fps of demo video output')
    args = parser.parse_args()
    return args

def frame_extraction(video_path,output):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join(output, osp.basename(osp.splitext(video_path)[0]))
    # target_dir = osp.join('./tmp','spatial_skeleton_dir')
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    while flag:
        frames.append(frame)
        #add frame path
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)
        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()
    return frame_paths, frames

def main():
    args = parse_args()
    # video_folder = '/home/duclam/Documents/dataset_action/Le2i_FDD_fall/Home/Videos'
    # for filename in sorted(glob.glob(args.folder_video + '/*.avi')):
    #     frame_paths, original_frames = frame_extraction(filename,args.output)
    #     num_frame = len(frame_paths)
    #     h, w, _ = original_frames[0].shape
    frame_paths, original_frames = frame_extraction(args.video,args.output)
    num_frame = len(frame_paths)
if __name__ == '__main__':
    main()