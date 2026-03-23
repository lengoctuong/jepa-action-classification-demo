# V-JEPA: Video Joint Embedding Predictive Architecture (with SSv2 Action Classification Demo)

Official PyTorch codebase for the *video joint-embedding predictive architecture*, V-JEPA, a method for self-supervised learning of visual representations from video.

*Note: This fork adds an interactive Web Demo application (Gradio) and detailed Benchmark scripts on the Something-Something V2 (SSv2) dataset based on pre-trained weights (Pretrained & Attentive Probe).*

## 1. Setup & Installation

System requirements: Linux/Windows operating system, Python 3.9+, and a CUDA-supported GPU.

### Initialize Conda environment

```bash
conda create -n jepa python=3.9 pip
conda activate jepa
python setup.py install
```

### Install additional libraries for the Fork

Required libraries for running the Web App (Gradio), video processing (OpenCV, MoviePy), and calculating Metrics (Scikit-learn, Pandas):

```bash
pip install gradio opencv-python scikit-learn pandas seaborn matplotlib tqdm moviepy
```

## 2. Data & Weights Preparation

### Download Model Weights

This fork is configured by default to use the **ViT-H/16** architecture (384x384 resolution) to achieve optimal accuracy on the SSv2 dataset. You need to download both the Encoder (frozen) and the Classifier layer (Attentive Probe).

```bash
# Download V-JEPA ViT-H/16 (384x384) Backbone
wget https://dl.fbaipublicfiles.com/jepa/vith16-384/vith16-384.pth.tar

# Download SSv2 Attentive Probe Classifier
wget https://dl.fbaipublicfiles.com/jepa/vith16-384/ssv2-probe.pth.tar
```

### Download SSv2 Test dataset from Kaggle

The dataset used for benchmarking is the Test set of Something-Something V2.

1.  Ensure you have configured the Kaggle API (`~/.kaggle/kaggle.json`).
2.  Download and extract the video data (Replace `<kaggle-dataset-id>` with the actual ID of the SSv2 dataset on Kaggle that you are using):

```bash
kaggle datasets download -d <kaggle-dataset-id>
unzip <kaggle-dataset-id>.zip -d ssv2_test_data/
```

3.  Ensure the directory structure containing Ground Truth Labels: `20bn-something-something-download-package-labels/labels/test-answers.csv` is available in the repository (can be downloaded from the [Labels](https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads) section of Qualcomm).

## 3. Running Scripts

### Run Web Demo App (Gradio)

The application uses the exact **Multi-view (16x2x3)** protocol pipeline from Meta FAIR, allowing video uploads or webcam usage for real-time action recognition.

```bash
python app.py
```

The interface will be hosted at `http://127.0.0.1:7860/` (supports Public Share Link).

### Inference & Benchmarking

**Execute Single-view Inference (Optimized for speed):**
Uses Center Crop and a single central clip.

```bash
python eval_ssv2_single_view.py
```

*Output:* `vjepa_preds_single_view.csv`

**Execute Multi-view Inference (Optimized for accuracy):**
Uses the 16x2x3 protocol (6 tensors/video). Note: Requires high GPU VRAM (Recommended 40GB+ for Batch Size 16).

```bash
python eval_ssv2_multiview.py
```

*Output:* `vjepa_preds_multiview.csv`

### Calculate Metrics (Precision, Recall, F1-Score)

After generating the prediction file (CSV), compare it with the Ground Truth file to output a detailed report for each action class (Macro/Micro Average).

```bash
python calculate_metrics.py
```

*(Note: Please point to the correct `PRED_CSV` variable path inside the `calculate_metrics.py` file corresponding to the results you just inferred).*

## 4. Source Code Structure (Extended)

```text
.
├── app.py                      # [NEW] Web App Demo for action recognition using Gradio (Official Pipeline)
├── eval_ssv2_single_view.py    # [NEW] Inference script (Single-view protocol)
├── eval_ssv2_multiview.py      # [NEW] Inference script (Multi-view 16x2x3 protocol)
├── calculate_metrics.py        # [NEW] Script for calculating Precision, Recall, F1-Score & Accuracy
├── ssv2_classes.json           # [NEW] List of 174 SSv2 labels
├── 20bn-something...labels/    # [NEW] Directory containing Ground Truth CSV
├── app/                        # [ORIGINAL] Training loops
├── evals/                      # [ORIGINAL] Evaluation modules
├── src/                        # [ORIGINAL] Model architectures (ViT, JEPA, etc.)
└── configs/                    # [ORIGINAL] YAML configs
```

-----

*(The section below retains the original documentation from Meta FAIR)*

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
