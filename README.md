# V-JEPA: Video Joint Embedding Predictive Architecture (with SSv2 Action Classification Demo)

Official PyTorch codebase for the *video joint-embedding predictive architecture*, V-JEPA, a method for self-supervised learning of visual representations from video.

*Lưu ý: Fork này bổ sung thêm ứng dụng Web Demo trực quan (Gradio) và các kịch bản chạy Benchmark chi tiết trên tập dữ liệu Something-Something V2 (SSv2) dựa trên các trọng số đã được huấn luyện trước (Pretrained & Attentive Probe).*

## 1\. Môi trường và Cài đặt (Setup & Installation)

Yêu cầu hệ thống: Hệ điều hành Linux/Windows, Python 3.9+ và GPU hỗ trợ CUDA.

### Khởi tạo môi trường Conda

```bash
conda create -n jepa python=3.9 pip
conda activate jepa
python setup.py install
```

### Cài đặt các thư viện bổ sung cho Fork

Các thư viện cần thiết để chạy Web App (Gradio), xử lý video (OpenCV, MoviePy) và tính toán Metrics (Scikit-learn, Pandas):

```bash
pip install gradio opencv-python scikit-learn pandas seaborn matplotlib tqdm moviepy
```

## 2\. Chuẩn bị Dữ liệu và Trọng số (Data & Weights Preparation)

### Tải trọng số Mô hình (Model Weights)

Fork này mặc định cấu hình sử dụng kiến trúc **ViT-H/16** (độ phân giải 384x384) để đạt độ chính xác tối ưu nhất trên tập SSv2. Bạn cần tải cả Encoder (đóng băng) và lớp Classifier (Attentive Probe).

```bash
# Tải V-JEPA ViT-H/16 (384x384) Backbone
wget https://dl.fbaipublicfiles.com/jepa/vith16-384/vith16-384.pth.tar

# Tải SSv2 Attentive Probe Classifier
wget https://dl.fbaipublicfiles.com/jepa/vith16-384/ssv2-probe.pth.tar
```

### Tải tập dữ liệu SSv2 Test từ Kaggle

Tập dữ liệu dùng để benchmark là tập Test của Something-Something V2.

1.  Đảm bảo bạn đã cấu hình Kaggle API (`~/.kaggle/kaggle.json`).
2.  Tải và giải nén dữ liệu video (Thay `<kaggle-dataset-id>` bằng ID thực tế của bộ dataset SSv2 trên Kaggle mà bạn sử dụng):

<!-- end list -->

```bash
kaggle datasets download -d <kaggle-dataset-id>
unzip <kaggle-dataset-id>.zip -d ssv2_test_data/
```

3.  Đảm bảo cấu trúc thư mục chứa Ground Truth Labels: `20bn-something-something-download-package-labels/labels/test-answers.csv` đã có sẵn trong repository (có thể download từ mục [Labels](https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads) của Qualcomm).

## 3\. Chạy Ứng dụng và Thực nghiệm (Running Scripts)

### Chạy Ứng dụng Web Demo (Gradio)

Ứng dụng sử dụng luồng xử lý giao thức **Multi-view (16x2x3)** chuẩn xác của Meta FAIR, cho phép upload video hoặc dùng webcam để nhận diện hành động theo thời gian thực.

```bash
python app.py
```

Giao diện sẽ được host tại `http://127.0.0.1:7860/` (hỗ trợ Public Share Link).

### Đánh giá Mô hình (Inference & Benchmarking)

**Thực thi Single-view Inference (Tối ưu tốc độ):**
Sử dụng Center Crop và 1 đoạn clip trung tâm duy nhất.

```bash
python eval_ssv2_single_view.py
```

*Đầu ra:* `vjepa_preds_single_view.csv`

**Thực thi Multi-view Inference (Tối ưu độ chính xác):**
Sử dụng giao thức 16x2x3 (6 tensors/video). Lưu ý: Yêu cầu VRAM GPU cao (Khuyến nghị 40GB+ cho Batch Size 16).

```bash
python eval_ssv2_multiview.py
```

*Đầu ra:* `vjepa_preds_multiview.csv`

### Tính toán Metrics (Precision, Recall, F1-Score)

Sau khi có file dự đoán (CSV), tiến hành đối chiếu với file Ground Truth để xuất báo cáo chi tiết cho từng lớp hành động (Macro/Micro Average).

```bash
python calculate_metrics.py
```

*(Ghi chú: Hãy trỏ đúng đường dẫn biến `PRED_CSV` bên trong file `calculate_metrics.py` tương ứng với kết quả bạn vừa infer).*

## 4\. Cấu trúc Source Code (Mở rộng)

```text
.
├── app.py                      # [MỚI] Web App Demo nhận diện hành động bằng Gradio (Official Pipeline)
├── eval_ssv2_single_view.py    # [MỚI] Inference script (Single-view protocol)
├── eval_ssv2_multiview.py      # [MỚI] Inference script (Multi-view 16x2x3 protocol)
├── calculate_metrics.py        # [MỚI] Script tính toán Precision, Recall, F1-Score & Accuracy
├── ssv2_classes.json           # [MỚI] Danh sách 174 nhãn SSv2
├── 20bn-something...labels/    # [MỚI] Thư mục chứa Ground Truth CSV
├── app/                        # [GỐC] Training loops
├── evals/                      # [GỐC] Evaluation modules
├── src/                        # [GỐC] Model architectures (ViT, JEPA, etc.)
└── configs/                    # [GỐC] YAML configs
```

-----

*(Phần dưới đây giữ nguyên tài liệu gốc từ Meta FAIR)*

## Method

V-JEPA pretraining is based solely on an unsupervised feature prediction objective, and does not utilize pretrained image encoders, text, negative examples, human annotations, or pixel-level reconstruction.

\<img src="[https://github.com/facebookresearch/jepa/assets/7530871/72df7ef0-2ef5-48bb-be46-27963db91f3d](https://github.com/facebookresearch/jepa/assets/7530871/72df7ef0-2ef5-48bb-be46-27963db91f3d)" width=40%\>
\&emsp;\&emsp;\&emsp;\&emsp;\&emsp;
\<img src="[https://github.com/facebookresearch/jepa/assets/7530871/f26b2e96-0227-44e2-b058-37e7bf1e10db](https://github.com/facebookresearch/jepa/assets/7530871/f26b2e96-0227-44e2-b058-37e7bf1e10db)" width=40%\>

## Visualizations

As opposed to generative methods that have a pixel decoder, V-JEPA has a predictor that makes predictions in latent space.
We train a conditional diffusion model to decode the V-JEPA feature-space predictions to interpretable pixels; the pretrained V-JEPA encoder and predictor networks are kept frozen in this process.
The decoder is only fed the representations predicted for the missing regions of the video, and does not have access to the unmasked regions of the video.

\<img src="[https://github.com/facebookresearch/jepa/assets/7530871/8bb5e338-0db8-4532-ba6f-fc62729acc26](https://github.com/facebookresearch/jepa/assets/7530871/8bb5e338-0db8-4532-ba6f-fc62729acc26)" width=90%\>

## MODEL ZOO

#### Pretrained models

*(Details omitted for brevity - See original FAIR repository for complete Model Zoo links for IN1K, Places205, iNat21, K400)*

## License

See the [LICENSE](https://www.google.com/search?q=./LICENSE) file for details about the license under which this code is made available.

## Citation

If you find this repository useful in your research, please consider giving a star :star: and a citation

```bibtex
@article{bardes2024revisiting,
  title={Revisiting Feature Prediction for Learning Visual Representations from Video},
  author={Bardes, Adrien and Garrido, Quentin and Ponce, Jean and Rabbat, Michael, and LeCun, Yann and Assran, Mahmoud and Ballas, Nicolas},
  journal={arXiv:2404.08471},
  year={2024}
}
```
