import os
import cv2
import random
import shutil
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from img_enhance import LowLightEnhancer


# sample function
def sampling_video(video_path, sampled_video_path=None, num_frames=20, sampling_type='uniform'):
    # initialization and determine whether the file needs to be saved
    is_save = None
    out = None
    sampled_frame_indices = []
    if sampled_video_path:
        is_save = True

    # read video file
    capture = cv2.VideoCapture(video_path)

    # get total frames and fps
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    if sampling_type == 'uniform':
        interval = total_frames // num_frames
        outside_frames = total_frames - interval * (num_frames - 1)
        start_frame = outside_frames // 2
        sampled_frame_indices = [start_frame + interval * i for i in range(num_frames)]

    elif sampling_type == 'random':
        sampled_frame_indices = random.sample(range(total_frames), num_frames)

    # create a VideoWriter object
    if is_save:
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        out = cv2.VideoWriter(sampled_video_path, fourcc, fps, (int(capture.get(3)), int(capture.get(4))))

    # loop every frame of the video
    frames = []
    current_frame = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        # uniform/random sampling
        if current_frame in sampled_frame_indices:
            if is_save:
                out.write(frame)
            frames.append(frame)
        current_frame += 1

    # release the VideoCapture
    capture.release()
    if is_save:
        out.release()
    cv2.destroyAllWindows()

    # (frames/time, Height, Width, channels) to (frames/time,channels,Height,Width)
    return np.array(frames).transpose((0, 3, 1, 2))


# process videos and copy
def process_copy(source_folder, target_folder, target_ext, func, *args):
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith("." + target_ext):
                source_path = os.path.join(root, file)
                target_path = source_path.replace(source_folder, target_folder)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                func(source_path, target_path, *args)
            else:
                source_path = os.path.join(root, file)
                target_path = source_path.replace(source_folder, target_folder)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copy(source_path, target_path)


# can also use torchvision.transforms
def video_normalization(frames, mean, std):
    mean = np.array(mean).reshape((1, 3, 1, 1))
    std = np.array(std).reshape((1, 3, 1, 1))
    # normalize video
    normalized_video = (frames / 255.0 - mean) / std

    return normalized_video


# accuracy of several classifiers
def classifier_hub(x_train, y_train, x_test, y_test):
    classifiers = {
        'SVM': SVC(kernel='linear', C=1.0),
        'Naive Bayes': GaussianNB(),
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=100, solver='liblinear'),
        'Random Forest': RandomForestClassifier(random_state=37),
        'XGBoost': XGBClassifier(),
    }
    results = {'Classifier': [], 'Accuracy': []}
    for clf_name, clf in classifiers.items():
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        results['Classifier'].append(clf_name)
        results['Accuracy'].append(accuracy)

    results_df = pd.DataFrame(results)
    print(f'results of classifiers:\n {results_df}')
    return results, results_df


def calculate_mean_std(source_dir, enhance_pipeline):
    vpixel = np.array([], dtype=np.float32).reshape(0, 3)
    lle = LowLightEnhancer(None, None, enhance_pipeline)
    count = 0
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".mp4"):
                if count == 10:
                    break
                count += 1
                source_path = os.path.join(root, file)
                cap = cv2.VideoCapture(source_path)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = np.array(frame)
                    enhanced_frame = lle.enhance_frame(frame) / 255.0
                    vpixel = np.vstack((vpixel, enhanced_frame.reshape(-1, 3)))
                cap.release()
    mean = np.mean(vpixel, axis=0)
    std = np.std(vpixel, axis=0)

    return mean, std


if __name__ == "__main__":
    # ***For Debugging***
    # pick a video to sample and normalize
    video_path = 'data/train/Jump/Jump_8_1.mp4'
    uniform_sampled_video_path = 'single_sampled_video/Jump_8_1_uniform.mp4'
    random_sampled_video_path = 'single_sampled_video/Jump_8_1_random.mp4'
    num_frames = 20
    mean = [0.07, 0.07, 0.07]
    std = [0.1, 0.09, 0.08]
    source_dir = './data/train'
    enhance_pipeline = ['bright_contrast_adjust']
    # uniform/random sampling
    uniform_sampled_video = sampling_video(video_path, uniform_sampled_video_path, num_frames, 'uniform')
    random_sampled_video = sampling_video(video_path, random_sampled_video_path, num_frames, 'random')
    # normalization
    normed_uniform_sampled_video = video_normalization(uniform_sampled_video, mean, std)
    # calculate mean and std
    mean, std = calculate_mean_std(source_dir, enhance_pipeline)
    print(mean, std)
