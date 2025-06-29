import os
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms

class SignLanguageDataset(Dataset):
    def __init__(self, csv_path, video_dir, num_frames=16, transform=None):
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.transform = transform

        # CSV 파일 읽기
        self.annotations = pd.read_csv(csv_path)

        # 문자열 라벨을 숫자 인덱스로 매핑
        self.label_list = sorted(self.annotations['label'].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.label_list)}
        self.num_classes = len(self.label_list)


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        filename = row['filename']  # ex: KETI_SL_0000000001.MOV
        label_str = row['label']
        label = self.label_to_idx[label_str]  # 문자열 라벨 -> 정수 인덱스


        video_path = os.path.join(self.video_dir, filename)
        if not os.path.exists(video_path):
            if "filename" not in video_path:
                print(f"🔴 파일 없음: {video_path}")
            # 예외 발생시키거나, 더미 프레임 생성 후 반환
            dummy_frame = torch.zeros(3, 224, 224)  # C,H,W
            video_tensor = torch.stack([dummy_frame] * self.num_frames)
            return video_tensor, label

        frames = self._load_video_frames(video_path)
        if len(frames) == 0:
            print(f"🔴 영상 프레임 로딩 실패: {video_path}")
            dummy_frame = torch.zeros(3, 224, 224)
            video_tensor = torch.stack([dummy_frame] * self.num_frames)
            return video_tensor, label

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        video_tensor = torch.stack(frames)  # (T, C, H, W)
        return video_tensor, label

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = self._sample_indices(total_frames)

        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                continue
            if i in frame_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
        cap.release()
        frames = frames[:self.num_frames]  # safety
        return [transforms.ToTensor()(f) for f in frames]

    def _sample_indices(self, total):
        if total <= self.num_frames:
            return list(range(total))
        interval = total // self.num_frames
        return [i * interval for i in range(self.num_frames)]
