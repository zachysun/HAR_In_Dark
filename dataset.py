import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
from utils import sampling_video
from img_enhance import LowLightEnhancer
from optical_flow_dense import video2opflow


class DarkVideoDataset(Dataset):
    def __init__(self, source_dir, data_type='rgb', mode='train', transform=None, LowLightEnhancer=None,
                 sampling_type='uniform', num_frames=20, split_ratio=0.8, is_val=False, seed=37):

        self.source_dir = source_dir
        self.mode = mode
        self.transform = transform
        self.LowLightEnhancer = LowLightEnhancer
        self.sampling_type = sampling_type
        self.num_frames = num_frames
        self.mapping_table = self._load_mapping_table()
        self.data_type = data_type
        if self.mode in ['train', 'validate']:
            self.data = self._load_data(f'{mode}.txt')
        elif self.mode == 'train_val':
            train_val_data = self._load_data('train.txt')
            train_size = int(len(train_val_data) * split_ratio)
            validate_size = len(train_val_data) - train_size
            train_data, validate_data = random_split(train_val_data, [train_size, validate_size],
                                                     generator=torch.Generator().manual_seed(seed))
            self.data = validate_data if is_val else train_data
        else:
            raise ValueError(f"Unsupported mode: {self.mode}, choose from ['train', 'validate', 'train_val']")

    def _load_mapping_table(self):
        mapping_table_path = os.path.join(self.source_dir, 'mapping_table_24.txt')
        with open(mapping_table_path, 'r') as file:
            mapping_table = dict(line.strip().split('\t') for line in file.readlines())
        return mapping_table

    def _load_data(self, filename):
        data_path = os.path.join(self.source_dir, filename)
        with open(data_path, 'r') as file:
            data = [line.strip().split('\t') for line in file.readlines()]
        return data

    # sampling video
    def _load_frames(self, video_path):
        sampled_frames = sampling_video(video_path, None, self.num_frames,
                                        self.sampling_type)
        # low light image enhancement
        if self.LowLightEnhancer:
            sampled_frames = torch.stack([torch.tensor(self.LowLightEnhancer(frame)) for frame in sampled_frames])
            sampled_frames = sampled_frames.permute(0, 3, 1, 2)

        return sampled_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        subdir = ''
        if self.mode in ['train', 'train_val']:
            subdir = 'train'
        elif self.mode == 'validate':
            subdir = 'validate'
        video_path = os.path.join(self.source_dir, subdir, self.data[idx][2])
        label = int(self.data[idx][1])
        # load frames
        frames = self._load_frames(video_path)
        # extract optical flows
        if self.data_type == 'rgb':
            frames = frames
        elif self.data_type == 'flow':
            # (frames, channels, Height, Width) to (frames, Height, Width, channels)
            # (frames, Height, Width, channels)  to (channels, frames, Height, Width)
            frames = video2opflow(frames.permute(0, 2, 3, 1))
            # (channels, frames, Height, Width) to (frames, channels, Height, Width)
            frames = torch.tensor(frames).permute(1, 0, 2, 3)
        else:
            raise ValueError(f"Unsupported date type: {self.data_type}, choose from ['rgb', 'flow']")

        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])
        # (frames, channels, Height, Width) to (channels, frames, Height, Width)
        return frames.permute(1, 0, 2, 3), label


if __name__ == "__main__":
    # ***For Debugging***
    # rgb
    transform = transforms.Compose([
        transforms.ToPILImage(),
        LowLightEnhancer(None, None, ['gamma_correction']),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.07, 0.07, 0.07], std=[0.1, 0.09, 0.08])
    ])

    train_dataset = DarkVideoDataset(source_dir='data', mode='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)

    val_dataset = DarkVideoDataset(source_dir='data', mode='validate', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print('train rgb data batch...')
    for train_data in train_loader:
        frames, labels = train_data
        print(frames.shape)
        print(labels.shape)

    print('validate rgb data batch...')
    for val_data in val_loader:
        frames, labels = val_data
        print(frames.shape)
        print(labels.shape)

    print('***************')
    # flow
    transform_flow = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset_flow = DarkVideoDataset(source_dir='data', data_type='flow', mode='train',
                                          transform=transform_flow)
    train_loader_flow = DataLoader(train_dataset_flow, batch_size=16, shuffle=False)

    val_dataset_flow = DarkVideoDataset(source_dir='data', data_type='flow', mode='validate',
                                        transform=transform_flow)
    val_loader_flow = DataLoader(val_dataset_flow, batch_size=16, shuffle=False)

    print('train flow data batch...')
    for train_data in train_loader_flow:
        frames, labels = train_data
        print(frames.shape)
        print(labels.shape)

    print('validate flow data batch...')
    for val_data in val_loader_flow:
        frames, labels = val_data
        print(frames.shape)
        print(labels.shape)
