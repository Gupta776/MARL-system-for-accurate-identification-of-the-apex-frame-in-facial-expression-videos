import csv
import os
import shutil
import cv2
import torch
from facenet_pytorch import MTCNN
import pickle
import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from itertools import product

def extract_images(data_file_path, dataset_path, output_path):
    skipped_videos = []
    with open(data_file_path, newline='') as csvfile:
        data = csv.DictReader(csvfile)
        for row in data:
            onset_frame = int(row['OnsetF'])
            apex_frame1 = int(row['ApexF1'])
            emotion = row['Emotion']
            video_path = os.path.join(dataset_path, emotion, row['Filename'])
            if not os.path.isdir(video_path):
                skipped_videos.append((emotion, row['Filename']))
                continue
            output_dir = os.path.join(output_path, emotion, row['Filename'])
            os.makedirs(output_dir, exist_ok=True)
            for i, filename in enumerate(sorted(os.listdir(video_path))):
                frame_number = int(filename.split('.')[0].split('-')[1])
                if frame_number >= onset_frame and frame_number <= apex_frame1:
                    frame_path = os.path.join(video_path, filename)
                    if not os.path.isfile(frame_path):
                        skipped_videos.append((emotion, row['Filename'], frame_number))
                        continue
                    output_filename = f'frame_{frame_number:03d}.jpg'
                    shutil.copy(frame_path, os.path.join(output_dir, output_filename))
                    if frame_number == apex_frame1:
                        break
    return skipped_videos


def align_face(input_path, output_path, mtcnn):
    image = cv2.imread(input_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    aligned_face = mtcnn(image, save_path=output_path)

    if aligned_face is None:
        print(f"Face not detected in {input_path}")
        return False
    
    # aligned_face = aligned_face.permute(1, 2, 0).numpy()
    # aligned_face = cv2.convertScaleAbs(aligned_face, alpha=(255.0))  # Scale pixel values to range 0-255
    # aligned_face = cv2.resize(aligned_face, (image_size, image_size))
    # aligned_face_gray = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2GRAY)  # Convert the image to grayscale
    # cv2.imwrite(output_path, aligned_face_gray)
    return True

def process_dataset(input_root, output_root, image_size=224):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(device=device, margin = 20, image_size=image_size)

    for expression in os.listdir(input_root):
        expression_path = os.path.join(input_root, expression)
        output_expression_path = os.path.join(output_root, expression)
        os.makedirs(output_expression_path, exist_ok=True)

        for episode in os.listdir(expression_path):
            episode_path = os.path.join(expression_path, episode)
            output_episode_path = os.path.join(output_expression_path, episode)
            os.makedirs(output_episode_path, exist_ok=True)

            for frame in os.listdir(episode_path):
                frame_path = os.path.join(episode_path, frame)
                output_frame_path = os.path.join(output_episode_path, frame)

                align_face(frame_path, output_frame_path, mtcnn)
def compute_lbp_top_features(data_path, image_size=(224, 224)):
    lbp_top_dict = {}
    episodes = _load_episodes(data_path)

    for episode_path in episodes:
        frames = _get_frames(episode_path)
        reference_frame = _read_frame(frames[0], image_size)
        lbp_top_features = []

        for frame_path in frames[1:]:
            current_frame = _read_frame(frame_path, image_size)
            lbp_top_feature = compute_lbp_top(reference_frame, current_frame)
            lbp_top_features.append(lbp_top_feature)

        lbp_top_dict[episode_path] = np.array(lbp_top_features)

    return lbp_top_dict


def _load_episodes(data_path):
    episodes = []
    for expression in os.listdir(data_path):
        expression_path = os.path.join(data_path, expression)
        for episode in os.listdir(expression_path):
            episode_path = os.path.join(expression_path, episode)
            episodes.append(episode_path)
    return episodes


def _get_frames(episode_path):
    frames = []
    for frame in os.listdir(episode_path):
        frame_path = os.path.join(episode_path, frame)
        frames.append(frame_path)
    return frames


def _read_frame(frame_path, image_size):
    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    frame = cv2.resize(frame, image_size)
    return frame


def compute_lbp_top(reference_frame, current_frame, radius=2, n_points=8):
    lbp_reference = local_binary_pattern(reference_frame, n_points, radius)
    lbp_current = local_binary_pattern(current_frame, n_points, radius)

    lbp_top = np.abs(lbp_reference - lbp_current)
    lbp_top = lbp_top / (np.linalg.norm(lbp_top) + 1e-8)

    return lbp_top.flatten()



def save_lbp_top_features(lbp_top_dict, filename):
    with open(filename, 'wb') as f:
        pickle.dump(lbp_top_dict, f)
        


if __name__ == '__main__':
    # skipped_videos = extract_images('CASME.csv', 'CASME/CASME', 'CASME/Episodes')
    # print('Skipped videos:')
    # for video in skipped_videos:
    #     print(video)  
    input_root = "Episodes"
    output_root = "face_aligned_episodes"

    process_dataset(input_root, output_root, image_size=224) 

    # data_path = "face_aligned_episodes"
    # lbp_top_dict = compute_lbp_top_features(data_path)
    # output_filename = "lbp_top_features.pkl"
    # save_lbp_top_features(lbp_top_dict, output_filename)