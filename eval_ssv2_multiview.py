import os
import glob
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys
from PIL import Image

cv2.setNumThreads(0)
sys.path.append(os.path.join(os.getcwd()))

# --- IMPORT MODULES CHUẨN TỪ META FAIR ---
from src.models.vision_transformer import vit_huge
from src.models.attentive_pooler import AttentiveClassifier 
from evals.video_classification_frozen.utils import make_transforms, ClipAggregation

def clean_state_dict(state_dict):
    """Làm sạch prefix của state_dict (nếu có do DDP)"""
    new_dict = {}
    for k, v in state_dict.items():
        k = k.replace('module.', '').replace('backbone.', '')
        new_dict[k] = v
    return new_dict

# ---------------------------------------------------------
# 1. Dataset Multi-view sử dụng Transform chuẩn
# ---------------------------------------------------------
class SSv2DatasetMultiViewOfficial(Dataset):
    def __init__(self, video_paths, transform, num_frames=16, frame_step=4):
        self.video_paths = video_paths
        self.transform = transform
        self.num_frames = num_frames
        self.frame_step = frame_step

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        
        total_frames = len(frames)
        if total_frames == 0:
            return video_id, None

        clip_length = self.num_frames * self.frame_step
        
        # Lấy 2 mốc thời gian (Đầu và Cuối)
        temporal_starts = [0, 0] if total_frames <= clip_length else [0, total_frames - clip_length]
        
        clips = [] # Lưu trữ 2 temporal clips, mỗi clip chứa 3 spatial views
        
        for start_idx in temporal_starts:
            clip_pil_frames = []
            
            # Lấy 16 frames và chuyển sang RGB PIL Image
            for i in range(self.num_frames):
                f_idx = min(start_idx + i * self.frame_step, total_frames - 1)
                frame_rgb = cv2.cvtColor(frames[f_idx], cv2.COLOR_BGR2RGB)
                clip_pil_frames.append(Image.fromarray(frame_rgb))
            
            # Transform chuẩn sinh ra list chứa 3 Tensors: [C, T, H, W]
            spatial_views = self.transform(clip_pil_frames) 
            clips.append(spatial_views)

        return video_id, clips

def collate_fn_official(batch):
    """
    Sắp xếp lại dữ liệu từ Dataset thành cấu trúc chuẩn cho ClipAggregation:
    Đầu vào từ Dataset: List của [video_id, clips]
      - clips: [temporal_1, temporal_2]
      - temporal_n: [spatial_1, spatial_2, spatial_3] (Tensor shape [C, T, H, W])
      
    Đầu ra mong đợi của ClipAggregation:
      - batched_clips: List gồm 2 phần tử (temporal)
      - batched_clips[0]: List gồm 3 phần tử (spatial)
      - batched_clips[0][0]: Tensor đã được stack qua Batch Dimension -> [B, C, T, H, W]
    """
    batch = list(filter(lambda x: x[1] is not None, batch))
    if len(batch) == 0: return [], None
    
    video_ids = [item[0] for item in batch]
    num_temporal_clips = len(batch[0][1])  # 2
    num_spatial_views = len(batch[0][1][0]) # 3
    
    batched_clips = []
    for t_idx in range(num_temporal_clips):
        batched_views = []
        for s_idx in range(num_spatial_views):
            # Gộp tensor của cùng một view qua toàn bộ batch
            view_tensor = torch.stack([item[1][t_idx][s_idx] for item in batch])
            batched_views.append(view_tensor)
        batched_clips.append(batched_views)
        
    return video_ids, batched_clips

# ---------------------------------------------------------
# 2. Vòng lặp Inference
# ---------------------------------------------------------
def run_batch_inference_official(dataset_folder, output_csv, encoder, classifier, eval_transform, device, batch_size=16, num_workers=12):
    search_pattern = os.path.join(dataset_folder, "*", "*.webm")
    all_video_files = glob.glob(search_pattern)
    
    processed_ids = set()
    if os.path.exists(output_csv):
        with open(output_csv, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith("video_id"):
                    processed_ids.add(line.split(',')[0])
    
    pending_videos = [v for v in all_video_files if os.path.splitext(os.path.basename(v))[0] not in processed_ids]
    
    print(f"Tổng số video: {len(all_video_files)}")
    print(f"Đã xử lý trước đó: {len(processed_ids)}")
    print(f"Cần xử lý: {len(pending_videos)}")
    
    if len(pending_videos) == 0: return

    dataset = SSv2DatasetMultiViewOfficial(pending_videos, transform=eval_transform)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn_official
    )
    
    with open(output_csv, 'a', encoding='utf-8') as f:
        if len(processed_ids) == 0: f.write("video_id,pred_class_id\n")
            
        for video_ids, batched_clips in tqdm(dataloader, desc="Official Multi-view Infer", unit="batch"):
            if batched_clips is None: continue
            
            # Chuyển toàn bộ list of lists of tensors lên thiết bị (GPU)
            for t_idx in range(len(batched_clips)):
                for s_idx in range(len(batched_clips[t_idx])):
                    batched_clips[t_idx][s_idx] = batched_clips[t_idx][s_idx].to(device, non_blocking=True)
            
            with torch.no_grad(), torch.amp.autocast('cuda'):
                # 1. Chạy qua Encoder (ClipAggregation tự động flatten batch)
                outputs = encoder(batched_clips) 
                
                # 2. Chạy qua Classifier
                # outputs: list(3) các spatial views, mỗi view là list(2) các temporal clips
                classifier_outs = [[classifier(ost) for ost in os] for os in outputs]
                
                # 3. Average Pooling xác suất từ toàn bộ 6 views (2 temporal * 3 spatial)
                num_views_total = len(classifier_outs) * len(classifier_outs[0])
                avg_probs = sum([sum([F.softmax(ost, dim=-1) for ost in os]) for os in classifier_outs]) / num_views_total
                
            pred_classes = torch.argmax(avg_probs, dim=-1)
            
            for vid, pred in zip(video_ids, pred_classes):
                f.write(f"{vid},{pred.item()}\n")
            f.flush() 
            
    print(f"\nHoàn tất! Kết quả được lưu tại: {output_csv}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_folder = "ssv2_test_data" 
    output_csv = "vjepa_preds_multiview.csv" 
    
    # ⚠️ Lưu ý VRAM: Batch size 16 * 6 views = 96 tensors [C, T, H, W] đưa vào GPU cùng lúc.
    # Tương đương ngốn khoảng 35-40GB VRAM. Bạn có thể nâng lên 24 hoặc 32 nếu H100 còn dư sức.
    BATCH_SIZE = 16  
    NUM_WORKERS = 12 
    
    # Khởi tạo Transform chuẩn
    print("Đang cấu hình Transform chuẩn (16x2x3)...")
    eval_transform = make_transforms(
        training=False,
        num_views_per_clip=3, 
        crop_size=384
    )
    
    print("Đang nạp mô hình Encoder (ViT-Huge)...")
    base_encoder = vit_huge(img_size=384, patch_size=16, in_chans=3, num_frames=16, tubelet_size=2)
    enc_state = torch.load("vith16-384.pth.tar", map_location="cpu")
    enc_state = enc_state.get('target_encoder', enc_state.get('encoder', enc_state))
    base_encoder.load_state_dict(clean_state_dict(enc_state), strict=True)
    
    # Đóng gói Encoder bằng ClipAggregation
    encoder = ClipAggregation(base_encoder, tubelet_size=2, attend_across_segments=False)
    encoder.to(device).eval()
    
    print("Đang nạp mô hình Classifier...")
    classifier = AttentiveClassifier(embed_dim=1280, num_classes=174, num_heads=16)
    cls_state = torch.load("ssv2-probe.pth.tar", map_location="cpu")
    cls_state = cls_state.get('classifier', cls_state.get('state_dict', cls_state))
    classifier.load_state_dict(clean_state_dict(cls_state), strict=True)
    classifier.to(device).eval()
    
    run_batch_inference_official(dataset_folder, output_csv, encoder, classifier, eval_transform, device, BATCH_SIZE, NUM_WORKERS)