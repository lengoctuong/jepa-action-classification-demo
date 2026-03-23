import os
import glob
import json
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys

# Tắt đa luồng của OpenCV để tránh xung đột với DataLoader workers của PyTorch
cv2.setNumThreads(0)

sys.path.append(os.path.join(os.getcwd()))
from src.models.vision_transformer import vit_huge
from src.models.attentive_pooler import AttentiveClassifier 

# ---------------------------------------------------------
# 1. Định nghĩa Dataset tuỳ chỉnh cho PyTorch
# ---------------------------------------------------------
class SSv2Dataset(Dataset):
    def __init__(self, video_paths, target_size=384, num_frames=16, frame_step=4):
        self.video_paths = video_paths
        self.target_size = target_size
        self.num_frames = num_frames
        self.frame_step = frame_step

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        
        # Đọc video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        clip_length = self.num_frames * self.frame_step
        
        start_idx = (total_frames - clip_length) // 2 if total_frames > clip_length else 0
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        
        for i in range(clip_length):
            ret, frame = cap.read()
            if not ret: break
            if i % self.frame_step == 0:
                frames.append(frame)
                if len(frames) == self.num_frames: break
        cap.release()
        
        while len(frames) < self.num_frames and len(frames) > 0:
            frames.append(frames[-1])
            
        if len(frames) == 0:
            return video_id, None # Báo lỗi nếu video hỏng

        # Tiền xử lý không gian (Spatial)
        processed = []
        for frame in frames:
            h, w = frame.shape[:2]
            scale = self.target_size / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            y1 = (new_h - self.target_size) // 2
            x1 = (new_w - self.target_size) // 2
            frame = frame[y1:y1+self.target_size, x1:x1+self.target_size]
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            processed.append(frame)
            
        tensor = np.array(processed)
        tensor = np.transpose(tensor, (3, 0, 1, 2))  
        tensor = torch.tensor(tensor, dtype=torch.float32)
        
        return video_id, tensor

# Hàm lọc các video bị lỗi không đọc được để không làm gián đoạn batch
def collate_fn_filter_none(batch):
    batch = list(filter(lambda x: x[1] is not None, batch))
    if len(batch) == 0: return [], None
    video_ids = [item[0] for item in batch]
    tensors = torch.stack([item[1] for item in batch])
    return video_ids, tensors

# ---------------------------------------------------------
# 2. Vòng lặp Inference chính
# ---------------------------------------------------------
def run_batch_inference(dataset_folder, output_csv, encoder, classifier, device, batch_size=16, num_workers=8, search_pattern=None):
    all_video_files = glob.glob(search_pattern)
    
    # Tính năng Auto-Resume: Đọc các video đã xử lý từ file CSV
    processed_ids = set()
    if os.path.exists(output_csv):
        with open(output_csv, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    processed_ids.add(line.split(',')[0])
    
    # Lọc ra các video chưa chạy
    pending_videos = [v for v in all_video_files if os.path.splitext(os.path.basename(v))[0] not in processed_ids]
    
    print(f"Tổng số video: {len(all_video_files)}")
    print(f"Đã xử lý trước đó: {len(processed_ids)}")
    print(f"Cần xử lý: {len(pending_videos)}")
    
    if len(pending_videos) == 0:
        print("Đã hoàn thành toàn bộ dataset!")
        return

    # Khởi tạo DataLoader
    dataset = SSv2Dataset(pending_videos)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True, # Tăng tốc độ chuyển data từ RAM sang VRAM
        collate_fn=collate_fn_filter_none
    )
    
    # Mở file ở chế độ 'a' (append - ghi nối tiếp)
    with open(output_csv, 'a', encoding='utf-8') as f:
        # Ghi header nếu file mới tạo
        if len(processed_ids) == 0:
            f.write("video_id,pred_class_id\n")
            
        for video_ids, batch_tensors in tqdm(dataloader, desc="Batch Inference", unit="batch"):
            if batch_tensors is None: continue
            
            batch_tensors = batch_tensors.to(device, non_blocking=True)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda'): # Chạy mixed-precision để tăng tốc độ trên H100
                    latent = encoder(batch_tensors)
                    logits = classifier(latent)
                    pred_classes = torch.argmax(logits, dim=-1)
            
            # Ghi kết quả vào file ngay lập tức
            for vid, pred in zip(video_ids, pred_classes):
                f.write(f"{vid},{pred.item()}\n")
            
            f.flush() # Bắt buộc HĐH ghi dữ liệu xuống ổ cứng
            
    print(f"\nHoàn tất! Kết quả được lưu tại: {output_csv}")

def clean_state_dict(state_dict):
    """Xóa prefix 'module.' hoặc 'backbone.' sinh ra do DistributedDataParallel (DDP)"""
    new_dict = {}
    for k, v in state_dict.items():
        k = k.replace('module.', '')
        k = k.replace('backbone.', '')
        new_dict[k] = v
    return new_dict

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_folder = "ssv2_test_data" 
    # dataset_folder = "samples" 
    output_csv = "vjepa_predictions_single_view.csv" 
    
    # Tham số cấu hình hiệu năng (Điều chỉnh tùy lượng VRAM và CPU core)
    BATCH_SIZE = 128   # H100 80GB có thể đẩy lên 32 hoặc 64
    NUM_WORKERS = 12  # Số luồng CPU dùng để đọc video
    
    print("Đang nạp mô hình...")
    encoder = vit_huge(img_size=384, patch_size=16, in_chans=3, num_frames=16, tubelet_size=2)
    enc_checkpoint = torch.load("vith16-384.pth.tar", map_location="cpu")
    enc_state = enc_checkpoint.get('target_encoder', enc_checkpoint.get('encoder', enc_checkpoint))
    # Nạp weights với strict=True để đảm bảo nạp thành công 100%
    encoder.load_state_dict(clean_state_dict(enc_state), strict=True)
    encoder.to(device).eval()

    classifier = AttentiveClassifier(embed_dim=1280, num_classes=174, num_heads=16)
    cls_checkpoint = torch.load("ssv2-probe.pth.tar", map_location="cpu")
    cls_state = cls_checkpoint.get('classifier', cls_checkpoint.get('state_dict', cls_checkpoint))
    # Nạp weights với strict=True
    classifier.load_state_dict(clean_state_dict(cls_state), strict=True)
    classifier.to(device).eval()
    
    run_batch_inference(dataset_folder, output_csv, encoder, classifier, device, BATCH_SIZE, NUM_WORKERS, search_pattern=os.path.join(dataset_folder, "*", "*.webm"))