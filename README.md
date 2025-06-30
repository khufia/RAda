# RAda: One Last Attention for your Vision-Language Model

<div align="left" style="margin:24px 0;">
  <img src="https://user-images.githubusercontent.com/74038190/212284115-f47cd8ff-2ffb-4b04-b5bf-4d1c14c0247f.gif"
       width="100%" height="4"/>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2506.05336"><img src="https://img.shields.io/badge/arXiv-Paper-brightgreen?style=flat-square" alt="arXiv"></a>
</p>

<p align="center">
  <a href="https://liangchen527.github.io/"><b>Liang Chen</b></a><sup>*</sup>, 
  <a href="https://github.com/khufia"><b>Ghazi Shazan Ahmad</b></a><sup>*</sup>, 
  <a href="https://scholar.google.com/citations?user=JlUDjukAAAAJ&hl=en"><b>Tianjun Yao</b></a><sup></sup>, 
  <a href="https://lingqiao-adelaide.github.io/lingqiaoliu.github.io//"><b>Lingqiao Liu</b></a><sup></sup>, 
  <a href="https://zhiqiangshen.com/o//"><b>Zhiqiang Shen</b></a><sup></sup>, 

</p>


<p align="center">
  <b>MBZUAI</b> · <b>The University of Adelaide</b>
</p>

<p align="center"><sup>*</sup>Equal Technical Contributions</p>

---

## 🆕 Latest Updates

- 📢 **June 2025**: Our paper has been accepted to ICCV 2025!



## 📊 Overview

<p align="center">
  <img src="assets/tease.jpg" width="70%" alt="RAda Framework">
</p>

Pretrained vision-language models (VLMs), such as CLIP, achieve remarkable zero-shot performance, yet their downstream potential hinges on effective fine-tuning. Most adaptation methods typically focus on refining representation from separate modalities (text or vision) but neglect the critical role of their fused representations in the decision-making process, \emph{\ie} rational matrix that drives the final prediction \cite{chen2023domain}. 
%
To bridge the gap, we propose a simple yet effective \textbf{R}ational \textbf{Ada}ptaion ({RAda}) to explicitly exploit the final fused representation during fine-tuning. RAda employs a learned mask, obtained from a lightweight attention layer attached at the end of a VLM, to dynamically calibrate the contribution of each element in the rational matrix, enabling targeted adjustments to the final cross-modal interactions without incurring costly modifications to intermediate features.
%
Experiments in different settings (\emph{\ie} updating, or freezing pretrained encoders in adaptation, and test-time training that can only access the unlabeled test data) show that RAda serves as a versatile fine-tuning technique, improving the baseline with minimal code and performing comparably against current arts in most settings.

---
## 🏆 Highlights
Key contributions of **RAda**:
1. Rational Adaptation for VLMs: RAda extends the classical rational matrix to vision‑language models, directly modeling and fine‑tuning the fused image–text interactions that drive the final prediction.

2. Minimal, Encoder‑Agnostic Design: RAda plugs in a single multi‑query attention layer atop frozen encoders to learn a lightweight mask over the rational matrix—no encoder changes, just a few lines of code.

3. Broad, Consistent Gains: We validate RAda across full fine‑tuning, efficient adapter‑only tuning, and test‑time training, showing reliable improvements and competitive performance in all settings.
---
## 🧠 Architecture

<p align="center">
  <img src="assets/rational.jpg" alt="VideoGLaMM Architecture">
</p>
Rational Matrix in the CLIP decision-making process for a given image, where predictions (i.e. logits) are computed by summing each column. It fuses text and visual features and pro-vides a fine-grained characterization of different predictions.

---

## 📈 Results


> |1| Evaluations in the FFT setting.
<p align="center">
  <img src="assets/fft.png" alt="Benchmark results">
</p>


> |2| Comparison with EFT methods in the base-to-new setting.
<p align="center">
  <img src="assets/eft.png" alt="Benchmark results">
</p>

> |3| Comparisons with representative TTT methods regarding DG performance in four distribution shifts
<p align="center">
  <img src="assets/ttt.png" alt="Benchmark results">
</p>




---

## 🔧 Running VideoMolmo 

### Environment setup

(1) Setup environment and PyTorch
```bash
git clone https://github.com/mbzuai-oryx/VideoMolmo
cd VideoMolmo/VideoMolmo
conda create -n .videomolmo python=3.10 -y
conda activate .videomolmo
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

(2) Setup Molmo
```bash
git clone https://github.com/allenai/molmo.git
cd molmo && pip install -e .[all] && cd .. # setup molmo requirements
pip install -r requirements.txt
```

(3) Setup SAM
```bash
python setup.py build_ext --inplace # build sam2
mkdir -p sam2_checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O sam2_checkpoints/sam2.1_hiera_large.pt
```

### 🔄 Inference

To run inference on the provided sample video:

```bash
python infer.py \
  --video_path ../examples/video_sample1 \
  --prompt "point to the person in red shirt" \
  --save_path "results"
```

Your video should be a folder with all the frames. Sample structure:
```
video_sample1/
├── frame_0001.jpg
├── frame_0002.jpg
├── frame_0003.jpg
└── ...
```

Output includes segmentation masks for each frame and a JSON file (`points.jsonl`) containing point coordinates.
```
reuslts/
├── video_sample1/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   ├── frame_0003.jpg
│   ├── points.jsonl
│   └── ...
└── ...
```
### Training and Evaluation 🚀

To be released soon! Stay tuned for updates.


## Todos

- [ ] Release training and evaluation scripts.
- [ ] Add support for additional datasets.
- [ ] Release dataset creation pipeline.


## Citation 📜

```bibtex
  @misc{ahmad2025videomolmospatiotemporalgroundingmeets,
      title={VideoMolmo: Spatio-Temporal Grounding Meets Pointing},
      author={Ghazi Shazan Ahmad and Ahmed Heakl and Hanan Gani and Abdelrahman Shaker and Zhiqiang Shen and Ranjay Krishna and Fahad Shahbaz Khan and Salman Khan},
      year={2025},
      eprint={2506.05336},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
url={https://arxiv.org/abs/2506.05336},
}
```

---

[<img src="assets/MBZUAI_logo.png" width="360" height="90">](https://mbzuai.ac.ae)
[<img src="assets/allenai_logo.png" width="300" height="85">](https://allenai.org/)
[<img src="assets/UW_logo.png" width="300">](https://www.washington.edu/)
