import time
from datetime import datetime
import cv2
import json
import os
import subprocess
import torch
import torch.nn.functional as F
import gradio as gr
import sys
from PIL import Image

cv2.setNumThreads(0)
sys.path.append(os.path.join(os.getcwd()))

# --- IMPORT HÀNG CHUẨN TỪ REPO V-JEPA ---
from src.models.vision_transformer import vit_huge
from src.models.attentive_pooler import AttentiveClassifier 
from evals.video_classification_frozen.utils import make_transforms, ClipAggregation

def clean_state_dict(state_dict):
    new_dict = {}
    for k, v in state_dict.items():
        k = k.replace('module.', '').replace('backbone.', '')
        new_dict[k] = v
    return new_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang khởi tạo Model V-JEPA V1 trên {device}...")

# 1. Nạp Labels
ssv2_classes_path = "ssv2_classes.json"
if not os.path.exists(ssv2_classes_path):
    subprocess.run(["wget", "https://huggingface.co/datasets/huggingface/label-files/resolve/d79675f2d50a7b1ecf98923d42c30526a51818e2/something-something-v2-id2label.json", "-O", ssv2_classes_path])
with open(ssv2_classes_path, "r", encoding='utf-8') as f:
    SSV2_CLASSES = json.load(f)

# 2. Khởi tạo Transforms chuẩn (Không augmentation, cắt 3 góc)
eval_transform = make_transforms(
    training=False,
    num_views_per_clip=3, 
    crop_size=384
)

# 3. Khởi tạo Encoder & Bọc bằng ClipAggregation
print("Đang nạp trọng số Encoder...")
base_encoder = vit_huge(img_size=384, patch_size=16, in_chans=3, num_frames=16, tubelet_size=2)
enc_state = torch.load("vith16-384.pth.tar", map_location="cpu")
enc_state = enc_state.get('target_encoder', enc_state.get('encoder', enc_state))
base_encoder.load_state_dict(clean_state_dict(enc_state), strict=True)

# Đóng gói y hệt file main.py
encoder = ClipAggregation(base_encoder, tubelet_size=2, attend_across_segments=False)
encoder.to(device).eval()

# 4. Khởi tạo Classifier
print("Đang nạp trọng số Classifier...")
classifier = AttentiveClassifier(embed_dim=1280, num_classes=174, num_heads=16)
cls_state = torch.load("ssv2-probe.pth.tar", map_location="cpu")
cls_state = cls_state.get('classifier', cls_state.get('state_dict', cls_state))
classifier.load_state_dict(clean_state_dict(cls_state), strict=True)
classifier.to(device).eval()

def process_video_chunk(video_path):
    if not video_path: return "❌ Lỗi: Không nhận được file.", None
    start_total_time = time.time()
    
    try:
        # Đọc toàn bộ video
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame) 
        cap.release()
        
        total_frames = len(frames)
        if total_frames == 0: return "❌ Lỗi: Video trống.", None

        # Thiết lập thông số SSv2
        target_frames = 16
        frame_step = 4
        num_temporal_views = 2
        clip_length = target_frames * frame_step
        
        # Lấy 2 mốc thời gian (Đầu và Cuối)
        temporal_starts = [0, 0] if total_frames <= clip_length else [0, total_frames - clip_length]
        
        clips = [] # Định dạng yêu cầu của ClipAggregation: list của temporal clips, mỗi clip là list của spatial views
        
        for start_idx in temporal_starts:
            clip_pil_frames = []
            
            # Lấy 16 frames và chuyển sang PIL (Định dạng mà video_transforms yêu cầu)
            for i in range(target_frames):
                idx = min(start_idx + i * frame_step, total_frames - 1)
                frame_rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
                clip_pil_frames.append(Image.fromarray(frame_rgb))
            
            # EvalVideoTransform trả về list chứa 3 tensors (Left, Center, Right)
            spatial_views = eval_transform(clip_pil_frames) 
            
            # Thêm Batch Dimension (B=1) và đẩy lên GPU
            spatial_views = [view.unsqueeze(0).to(device) for view in spatial_views]
            clips.append(spatial_views)

        # --- INFERENCE ---
        torch.cuda.synchronize()
        start_infer_time = time.time()
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            # ClipAggregation sẽ lo việc tách ghép tensors và gọi base_encoder
            outputs = encoder(clips) 
            
            # Lặp qua các góc nhìn Không gian và Thời gian để tính qua Classifier
            # (Đoạn này copy 1:1 từ main.py của Meta)
            classifier_outs = [[classifier(ost) for ost in os] for os in outputs]
            
            # Trung bình cộng xác suất (Average Pooling)
            avg_probs = sum([sum([F.softmax(ost, dim=1) for ost in os]) for os in classifier_outs]) / len(classifier_outs) / len(classifier_outs[0])
            avg_probs = avg_probs[0] * 100.0 # Lấy batch index 0
            
        top3_values, top3_indices = torch.topk(avg_probs, 3)
        
        torch.cuda.synchronize()
        infer_duration = time.time() - start_infer_time
        
        # Tạo Log
        recv_time_str = datetime.now().strftime('%H:%M:%S')
        total_frames_infer = num_temporal_views * 3 * target_frames # 96 frames
        speed_per_frame = (infer_duration * 1000) / total_frames_infer 
        
        result_text = f"🕒 Thời gian nhận Input: {recv_time_str}\n"
        result_text += f"📥 Tổng số Frames gốc (Video): {total_frames}\n"
        result_text += f"🛠️ Module: Sử dụng `EvalVideoTransform` và `ClipAggregation` gốc.\n\n"
        result_text += f"⏱️ Thời gian Inference (GPU): {infer_duration:.3f}s\n"
        result_text += f"⚡ Tốc độ xử lý: ~{speed_per_frame:.2f} ms/frame\n\n"
        
        result_text += "🎯 Nhận diện Hành động (Average Pooling 16x2x3):\n"
        for i, (prob, idx) in enumerate(zip(top3_values, top3_indices)):
            class_name = SSV2_CLASSES.get(str(idx.item()), "Unknown")
            result_text += f"  {i+1}. {class_name}: {prob.item():.2f}%\n"
            
        return result_text

    except Exception as e:
        return f"❌ Lỗi: {str(e)}"

# ==========================================
# 4. GIAO DIỆN GRADIO
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎥 Nhận diện Hành động V-JEPA V1 (Meta FAIR Official Modules)")
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(sources=["webcam", "upload"], label="1. Quay hoặc Tải video lên")
            submit_btn = gr.Button("🚀 Phân tích Video", variant="primary")
            
        with gr.Column(scale=1):
            result_output = gr.Textbox(label="2. Log Kỹ Thuật & Kết quả", lines=15)

    # Note: Bỏ phần render video trả về vì eval_transform của Meta xuất trực tiếp ra Tensor Normalize, 
    # việc đảo ngược nó lại (denormalize) không đem lại nhiều ý nghĩa như lúc viết chay.
    submit_btn.click(fn=process_video_chunk, inputs=[video_input], outputs=[result_output])

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")