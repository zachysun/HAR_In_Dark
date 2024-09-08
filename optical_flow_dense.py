"""
Modified form https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
"""
import torch
import numpy as np
import cv2 as cv


def frames2opflow(frames):
    # frames (frames/time, h, w, channels)
    optical_flows = []
    prvs = cv.cvtColor(frames[0], cv.COLOR_BGR2GRAY)

    for i in range(1, len(frames)):
        frame = frames[i]
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        flow = cv.calcOpticalFlowFarneback(prvs, gray, None, 0.5, 3, 15,
                                           3, 5, 1.2, 0)
        optical_flows.append(flow)
        prvs = gray

    return np.array(optical_flows).transpose(3, 0, 1, 2)


def video2opflow(inputs, optical_flow_path=None):
    if isinstance(inputs, str):
        cap = cv.VideoCapture(inputs)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
    elif isinstance(inputs, np.ndarray):
        frames = inputs
    elif isinstance(inputs, torch.Tensor):
        frames = np.array(inputs)
    else:
        raise ValueError("Unsupported input type")

    # (channels, frames/time, h, w)
    optical_flows = frames2opflow(frames)

    if optical_flow_path is not None:
        np.save(optical_flow_path, optical_flows)

    return optical_flows


def visualize_opflow(inputs):
    if isinstance(inputs, str):
        optical_flows = frames2opflow(inputs)
    elif isinstance(inputs, np.ndarray):
        optical_flows = inputs
    else:
        raise ValueError("Unsupported input type")

    optical_flows = optical_flows.transpose(1, 2, 3, 0)
    for flow in optical_flows:
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 1] = 255
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        cv.imshow('Optical Flow Visualization', rgb)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    video_path = "./data/validate/244.mp4"
    optical_flows = video2opflow(video_path)
    print(optical_flows.shape)
    visualize_opflow(optical_flows)
