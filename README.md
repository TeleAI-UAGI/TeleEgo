<div align="center">
  <h1>
    TeleEgo: <br> 
    Benchmarking Egocentric AI Assistants in the Wild
  </h1>

  <!-- 项目徽章 -->
  <p>
    <a href="https://huggingface.co/datasets/David0219/TeleEgo">
      <img alt="Hugging Face" src="https://img.shields.io/badge/HuggingFace-Dataset-orange">
    </a>
    <a href="https://arxiv.org/abs/2510.23981">
      <img alt="arXiv" src="https://img.shields.io/badge/ArXiv-2510.23981-b31b1b.svg">
    </a>
    <a href="https://teleai-uagi.github.io/TeleEgo/">
      <img alt="Page" src="https://img.shields.io/badge/Project Page-Link-green">
    </a>
    <a href="https://seline02.github.io/2025/11/10/TeleEgo-%E6%B5%81%E5%BC%8F%E5%85%A8%E6%A8%A1%E6%80%81%E7%AC%AC%E4%B8%80%E4%BA%BA%E7%A7%B0%E8%AF%84%E6%B5%8B%E5%9F%BA%E5%87%86/">
      <img alt="Blog" src="https://img.shields.io/badge/Blog-Post-blue">
    </a>
  </p>

  <img src="assets/teaser.png" alt="Teaser" style="width:80%; max-width:700px;">
  
  📢 **Note**：This project is still under active development, and the benchmark will be continuously maintained.  
</div>

---

<div align="left">

**If you find this project helpful, please give us a ⭐️ on GitHub for the latest update.**

</div>




## 📌 Introduction

**TeleEgo** is a comprehensive **omni benchmark** designed for **multi-person, multi-scene, multi-task, and multimodal long-term memory reasoning** in egocentric video streams.
It reflects realistic personal assistant scenarios where continuous egocentric video data is collected across hours or even days, requiring models to maintain and reason over **memory, understanding, and cross-memory reasoning**. **Omni** here means that TeleEgo covers the full spectrum of **roles, scenes, tasks, modalities, and memory horizons**, offering all-round evaluation for egocentric AI assistants.

**TeleEgo provides:**

- 🧠 **Omni-scale, diverse egocentric data** from 5 roles across 4 daily scenarios.
- 🎤 **Multi-modal annotations**: video, narration, and speech transcripts.
- ❓ **Fine-grained QA benchmark**: 3 cognitive dimensions, 12 subcategories.


---

## 📊 Dataset Overview

- **Participants**: 5 (balanced gender)
- **Scenarios**:
  - Work & Study
  - Lifestyle & Routines
  - Social Activities
  - Outings & Culture
- **Recording**: 3 days/participant (~14.4 hours each)
- **Modalities**:
  - Egocentric video streams
  - Speech & conversations
  - Narration and event descriptions

---

## 🧪 Benchmark Tasks

TeleEgo-QA evaluates models along **three main dimensions**:

1. **Memory**
   - Short-term / Long-term / Ultra-long Memory
   - Entity Tracking
   - Temporal Comparison & Interval

2. **Understanding**
   - Causal Understanding
   - Intent Inference
   - Multi-step Reasoning
   - Cross-modal Understanding

3. **Cross-Memory Reasoning**
   - Cross-temporal Causality
   - Cross-entity Relation
   - Temporal Chain Understanding

Each QA instance includes:

- Question type: Single-choice, Multi-choice, Binary, Open-ended

---

<!-- ## Key Advantages over Existing Benchmarks

* **Compared with EgoLife**: TeleEgo offers **omni-scenario coverage** (not restricted to a single shared environment), broader task diversity, fine-grained memory categories, multi-task trajectories, and difficulty levels.
* **Compared with M3-Agent / HourVideo**: TeleEgo is explicitly **omni-task and omni-modal**, focusing on **diagnostic memory evaluation**, cross-event reasoning, and multimodal grounding in **real-life egocentric settings**.

--- -->

## 🗂️ Repository Structure

```
TeleEgo/
│
├── teleego_data/                # Dataset samples / metadata
│   ├── outputs/                 # Output results
│   ├── QAs/                     # Question-Answer pairs
│   └── video_merged/            # Merged video files
├── weights/                     # Pre-trained weights (MiniCPM-o, Qwen2.5-Omni, ...)
├── evaluate_gemini25_pro.py     # Evaluation script for Gemini 2.5 Pro
├── evaluate_gpt_4o.py           # Evaluation script for GPT-4o
├── evaluate_minicpm_o.py        # Evaluation script for MiniCPM-o
├── evaluate_qwen25_omni.py      # Evaluation script for Qwen2.5-Omni
├── evaluate_qwen25_vl.py        # Evaluation script for Qwen2.5-VL
├── evaluate_videochat_online.py # Evaluation script for VideoChat
├── metrics.py                   # Evaluation metrics
├── utils.py                     # Utility functions
├── run.sh                       # Execution script
└── README.md                    # This file
```

## 🚀 Usage

### 📥 Dataset Setup

1. **Download the dataset** from Hugging Face: 🔗 [**TeleEgo Dataset**](https://huggingface.co/datasets/David0219/TeleEgo)
   
   Or Baidu Netdisk: 🔗 [**TeleEgo Dataset**](https://pan.baidu.com/s/1T8LxTbrWIYUDXZlJyuR5Ew?pwd=ay5q)

2. **Organize the dataset** in the following structure:
```
./TeleEgo/teleego_data/
├── QAs/                              # Question-Answer dataset
│   ├── merged_P1_A.json             # QA data for participant P1
│   ├── merged_P2_A.json             # QA data for participant P2
│   ├── merged_P3_A.json             # QA data for participant P3
│   ├── merged_P4_A.json             # QA data for participant P4
│   └── merged_P5_A.json             # QA data for participant P5
├── outputs/                          # Evaluation outputs
│   ├── gemini25_pro/                # Results for Gemini 2.5 Pro
│   ├── gpt-4o/                      # Results for GPT-4o
│   ├── minicpm_o/                   # Results for MiniCPM-o
│   ├── qwen25_omni/                 # Results for Qwen2.5-Omni
│   ├── qwen25_vl/                   # Results for Qwen2.5-VL
│   └── videochat-online/            # Results for VideoChat-Online
└── video_merged/                     # Merged long videos with timestamps
    ├── merged_P1.mp4                # P1's 3-day video merged into one file
    ├── merged_P2.mp4                # P2's 3-day video merged into one file
    ├── merged_P3.mp4                # P3's 3-day video merged into one file
    ├── merged_P4.mp4                # P4's 3-day video merged into one file
    ├── merged_P5.mp4                # P5's 3-day video merged into one file
    ├── timeline_P1.json             # P1's timestamp mapping file
    ├── timeline_P2.json             # P2's timestamp mapping file
    ├── timeline_P3.json             # P3's timestamp mapping file
    ├── timeline_P4.json             # P4's timestamp mapping file
    └── timeline_P5.json             # P5's timestamp mapping file
```

### 🔧 Environment Setup

Set up your environment according to the official requirements of the model you want to evaluate:

- **Qwen2.5-Omni**: Follow the [official Qwen2.5-Omni setup guide](https://github.com/QwenLM/Qwen2.5-Omni)
- **MiniCPM-o**: Follow the [official MiniCPM-o setup guide](https://github.com/OpenBMB/MiniCPM-o)
- **Qwen2.5-VL**: Follow the [official Qwen2.5-VL setup guide](https://github.com/QwenLM/Qwen2-VL)
- **VideoChat-Online**: Follow the [official VideoChat-Online setup guide](https://github.com/MCG-NJU/VideoChat-Online)
- **GPT-4o / Gemini 2.5 Pro**: Configure your API credentials in `run.sh`

### 🧪 Running Evaluations

To evaluate a model on a specific GPU, use the following command format:
```bash
sh run.sh  
```

**Examples:**
```bash
# Evaluate Qwen2.5-Omni on GPU 0
sh run.sh eval_qwen25_omni 0
```

**Available evaluation functions:**
- `eval_qwen25_omni` - Qwen2.5-Omni model
- `eval_qwen25_vl` - Qwen2.5-VL model  
- `eval_minicpm_o` - MiniCPM-o model
- `eval_videochat_online` - VideoChat-Online model
- `eval_gpt_4o` - GPT-4o (requires API key)
- `eval_gemini25_pro` - Gemini 2.5 Pro (requires API key)


### 📊 Computing Metrics

After evaluation, the results will be saved in `./teleego_data/outputs/<model_name>/`. To compute evaluation metrics:
```bash
python metrics.py
```

This will calculate performance metrics across all evaluation dimensions (Memory, Understanding, Cross-Memory Reasoning).

### 📤 Submit Results

Submit your results to our 🏆 [**Online Leaderboard**](https://programmergg.github.io/jrliu.github.io/#leaderboard).

---

<!-- ## Baselines
![Baseline 1](assets/res1.png)
![Baseline 2](assets/res2.png)
---

## 🤝 Collaborators

Thanks to these amazing people for contributing to the project:

<a href="https://github.com/rebeccaeexu">
  <img src="https://avatars.githubusercontent.com/rebeccaeexu" width="60px" style="border-radius:50%" />
</a>
<a href="https://github.com/DavisWANG0">
  <img src="https://avatars.githubusercontent.com/DavisWANG0" width="60px" style="border-radius:50%" />
</a>
<a href="https://github.com/H-oliday">
  <img src="https://avatars.githubusercontent.com/H-oliday" width="60px" style="border-radius:50%" />
</a>
<a href="https://github.com/Xiaolong-RRL">
  <img src="https://avatars.githubusercontent.com/Xiaolong-RRL" width="60px" style="border-radius:50%" />
</a>
<a href="https://github.com/Programmergg">
  <img src="https://avatars.githubusercontent.com/Programmergg" width="60px" style="border-radius:50%" />
</a>
<a href="https://github.com/yiheng-wang-duke">
  <img src="https://avatars.githubusercontent.com/yiheng-wang-duke" width="60px" style="border-radius:50%" />
</a>
<a href="https://github.com/cocowy1">
  <img src="https://avatars.githubusercontent.com/cocowy1" width="60px" style="border-radius:50%" />
</a>
<a href="https://github.com/chxy95">
  <img src="https://avatars.githubusercontent.com/chxy95" width="60px" style="border-radius:50%" />
</a> -->


## 📜 Citation

If you find our **TeleEgo** in your research, please cite:

```bibtex
@article{yan2025teleego,
  title={TeleEgo: Benchmarking Egocentric AI Assistants in the Wild},
  author={Yan, Jiaqi and Ren, Ruilong and Liu, Jingren and Xu, Shuning and Wang, Ling and Wang, Yiheng and Wang, Yun and Zhang, Long and Chen, Xiangyu and Sun, Changzhi and others},
  journal={arXiv preprint arXiv:2510.23981},
  year={2025}
}
```

## 🪪 License

This project is licensed under the **MIT License**.
Dataset usage is restricted under a **research-only license**.

---

<!-- ## References

* EgoLife: Towards Egocentric Life Assistant [\[arXiv:2503.03803\]](https://arxiv.org/abs/2503.03803)
* M3-Agent: Seeing, Listening, Remembering, and Reasoning [\[arXiv:2508.09736\]](https://arxiv.org/abs/2508.09736)
* HourVideo: 1-Hour Video-Language Understanding [\[arXiv:2411.04998\]](https://arxiv.org/abs/2411.04998) -->


## 📬 Contact

If you have any questions, please feel free to reach out: chxy95@gmail.com.

---

<div align="center">

TeleEgo is an Omni benchmark, a step toward building personalized AI assistants with true long-term memory, reasoning and decision-making in real-world wearable scenarios.

Made with ❤️ by the Ubiquitous AGI team at TeleAI.

</div>

<div align="center" style="margin-top: 10px;">
  <img src="assets/TeleAI.jpg" alt="TeleAI Logo" width="120px" />
  &nbsp;&nbsp;&nbsp;
  <img src="assets/TeleEgo.png" alt="TeleEgo Logo" width="120px" />
</div>
