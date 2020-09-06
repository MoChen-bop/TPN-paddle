import os
import numpy as np
import cv2


def extract_frames(video_dir, save_dir):
    label_names = os.listdir(video_dir)
    label_dict = {}
    index = 0
    for label_name in label_names:
        if label_name.startswith('.'):
            continue
        label_dict[label_name] = index
        index += 1

        videos_path = os.path.join(video_dir, label_name)
        videos = os.listdir(videos_path)
        videos = filter(lambda x: x.endswith('avi'), videos)
        for video in videos:
            print('processing video: ' + video)

            video_path = os.path.join(videos_path, video)
            video_name = video[:-4]
            save_jpg_dir = os.path.join(save_dir, 'rgb', label_name, video_name)
            # save_flow_dir = os.path.join(save_dir, 'flow', label_name, video_name)
            if not os.path.exists(save_jpg_dir):
                os.makedirs(save_jpg_dir)
            # if not os.path.exists(save_flow_dir):
            #     os.makedirs(save_flow_dir)

            cap = cv2.VideoCapture(video_path)
            frame_count = 1
            success, prev_frame = cap.read()

            while success:
                success, current_frame = cap.read() 

                if success:
                    save_jpg_path = os.path.join(save_jpg_dir, 'img_%d.jpg' % frame_count)
                    cv2.imwrite(save_jpg_path, current_frame)

            #         save_u_flow_path = os.path.join(save_flow_dir, video_name + '_u_%d.jpg' % frame_count)
            #         save_v_flow_path = os.path.join(save_flow_dir, video_name + '_v_%d.jpg' % frame_count)
            #         prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            #         curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            #         flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            #         flow = (flow + 15) * (255.0 / (2 * 15))
            #         flow = np.round(flow).astype(int)
            #         flow[flow >= 255] = 255
            #         flow[flow < 0] = 0
            #         cv2.imwrite(save_u_flow_path, flow[:, :, 0])
            #         cv2.imwrite(save_v_flow_path, flow[:, :, 1])
                
            #     prev_frame = current_frame
                frame_count += 1

            cap.release()
    np.save(os.path.join(save_dir, 'label_dict.npy'), label_dict)
    print(label_dict)


if __name__ == '__main__':
    extract_frames('/home/aistudio/TPN/data/HMDB_51/avi', '/home/aistudio/TPN/data/HMDB_51')

