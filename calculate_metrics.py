import json
import os
import re
from collections import defaultdict

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[\[\]]', '', text) 
    text = ' '.join(text.split())      
    return text

def calculate_metrics(pred_csv, gt_csv, classes_json):
    print("Đang nạp dữ liệu đối chiếu...\n")
    
    # 1. Nạp từ điển mapping
    with open(classes_json, 'r', encoding='utf-8') as f:
        labels_dict = json.load(f)
    id_to_text = {str(k): normalize_text(v) for k, v in labels_dict.items()}

    # 2. Nạp Ground Truth
    gt_dict = {}
    with open(gt_csv, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) >= 2:
                gt_dict[parts[0]] = normalize_text(parts[1])

    correct_top1 = 0
    total_valid = 0
    
    # 3. Khởi tạo dictionaries đếm TP, FP, FN
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    actual_counts = defaultdict(int) 

    # 4. Đọc file dự đoán và phân loại kết quả
    with open(pred_csv, 'r', encoding='utf-8') as f:
        header = f.readline() 
        for line in f:
            if not line.strip(): continue
            vid, pred_id = line.strip().split(',')
            
            pred_text = id_to_text.get(pred_id, "")
            
            if vid in gt_dict:
                total_valid += 1
                true_text = gt_dict[vid]
                
                actual_counts[true_text] += 1
                
                if pred_text == true_text:
                    correct_top1 += 1
                    tp[true_text] += 1
                else:
                    fn[true_text] += 1
                    if pred_text: 
                        fp[pred_text] += 1

    if total_valid == 0:
        print("Lỗi: Không có ID video nào khớp.")
        return

    # 5. Khởi tạo biến lưu trữ Macro/Micro
    macro_precision_sum = 0.0
    macro_recall_sum = 0.0
    macro_f1_sum = 0.0
    
    global_tp = 0
    global_fp = 0
    global_fn = 0

    class_metrics = []
    num_classes = len(actual_counts)

    # 6. Tính toán Per-class Metrics và tích lũy Macro/Micro
    for cls_name, total in actual_counts.items():
        c_tp = tp[cls_name]
        c_fp = fp[cls_name]
        c_fn = fn[cls_name]
        
        # Per-class metrics
        precision = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0
        recall = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Tích lũy cho Macro
        macro_precision_sum += precision
        macro_recall_sum += recall
        macro_f1_sum += f1
        
        # Tích lũy cho Micro
        global_tp += c_tp
        global_fp += c_fp
        global_fn += c_fn

        class_metrics.append((cls_name, total, c_tp, precision, recall, f1))

    # 7. Tính Macro Averages
    macro_precision = macro_precision_sum / num_classes
    macro_recall = macro_recall_sum / num_classes
    macro_f1 = macro_f1_sum / num_classes

    # 8. Tính Micro Averages
    micro_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
    micro_recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    # 9. In Bảng Tổng Kết
    print("=" * 105)
    print("[KẾT QUẢ TỔNG THỂ]")
    print("=" * 105)
    print(f"- Số video đối chiếu: {total_valid}")
    print(f"- Top-1 Accuracy: {(correct_top1 / total_valid) * 100:.2f}%\n")
    
    print(f"{'Metric Type':<20} | {'Precision':<15} | {'Recall':<15} | {'F1-Score':<15}")
    print("-" * 75)
    print(f"{'Macro-Average':<20} | {macro_precision*100:>14.2f}% | {macro_recall*100:>14.2f}% | {macro_f1*100:>14.2f}%")
    print(f"{'Micro-Average':<20} | {micro_precision*100:>14.2f}% | {micro_recall*100:>14.2f}% | {micro_f1*100:>14.2f}%")
    print("=" * 105)

    # 10. In Thống kê Chi tiết
    print("\n[THỐNG KÊ CHI TIẾT THEO CLASS (Sắp xếp theo F1-Score giảm dần)]")
    print("-" * 105)
    print(f"{'Tên Class (Hành động)':<42} | {'Tổng':<4} | {'TP':<4} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9}")
    print("-" * 105)

    class_metrics.sort(key=lambda x: (x[5], x[1]), reverse=True)

    for cls_name, total, c_tp, precision, recall, f1 in class_metrics:
        display_name = cls_name[:39] + "..." if len(cls_name) > 42 else cls_name
        print(f"{display_name:<42} | {total:<4} | {c_tp:<4} | {precision*100:>7.2f}% | {recall*100:>7.2f}% | {f1*100:>7.2f}%")

if __name__ == "__main__":
    PRED_CSV = "vjepa_predictions_single_view.csv" 
    # PRED_CSV = "vjepa_preds_multiview.csv" 
    GT_CSV = "20bn-something-something-download-package-labels/labels/test-answers.csv"
    CLASSES_JSON = "../vjepa2/ssv2_classes.json" 
    
    calculate_metrics(PRED_CSV, GT_CSV, CLASSES_JSON)