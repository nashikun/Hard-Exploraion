import cv2
import numpy as np
import torch


def process_frame(frame) -> torch.Tensor:
    # Crop it down to 84x84
    frame = frame[30:195]
    frame = cv2.resize(frame, (84, 84))
    # Turn it into grayscale image
    frame = np.mean(frame, axis=2).astype(np.uint8)
    return torch.from_numpy(frame).float()


def make_state(frames: list) -> torch.Tensor:
    x = []
    for frame in frames:
        x.append(process_frame(frame))
    return torch.from_numpy(np.stack(x, axis=0).reshape((4, 84, 84))).float()
