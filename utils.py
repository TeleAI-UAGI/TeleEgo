import re
import json
import math
import tqdm
import torch
import librosa
import pickle
import tempfile
import numpy as np
from PIL import Image
import soundfile as sf
from pathlib import Path
from moviepy import VideoFileClip
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Any, Tuple, Optional
import time
import threading
import bisect
import pdb
import signal
import os
from transformers import StoppingCriteria, StoppingCriteriaList
import sys
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import random

API_KEY = ""
API_VERSION = "2024-08-01-preview"
END_POINT = ""
ENGINE = "4o"

LLM_EVAL_CONFIG = {
    "enabled": True,
    "provider": "azure",
    "api_key": API_KEY,
    "azure_endpoint": END_POINT,
    "azure_api_version": API_VERSION,
    "azure_deployment": ENGINE,
    "temperature": 0.0,
    "timeout": 30,
    "prompt": "open_ended_cn_v1",
}


class TimeLimitStoppingCriteria(StoppingCriteria):
    def __init__(self, max_time_seconds: float):
        self.start_time = time.time()
        self.max_time_seconds = max_time_seconds

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return (time.time() - self.start_time) >= self.max_time_seconds




###############################################
# Time helpers
###############################################
def parse_hhmmss(ts: str) -> float:
    """将 HH:MM:SS 或 MM:SS 格式转换为秒数"""
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = "0", parts[0], parts[1]
    else:
        raise ValueError(f"Bad timestamp: {ts}")
    return int(h) * 3600 + int(m) * 60 + float(s)

###############################################
# Timeline helpers
###############################################
class TimelineIndex:
    """
    处理合并视频的时间映射
    将 "D1-09:31:38" 这样的标签转换为合并视频中的秒数
    """
    def __init__(self, segments, day_prefix="D"):
        self.segments = sorted(segments, key=lambda s: s["merged_offset_start_seconds"])
        self.day_prefix = day_prefix

    @staticmethod
    def _parse_day_label(label: str, day_prefix: str = "D") -> Optional[int]:
        """解析 D1-09:31:38 格式，返回绝对秒数（从第0天开始）"""
        if not isinstance(label, str):
            return None
        s = label.strip()
        # 统一符号
        s = (s.replace("：", ":").replace("—", "-").replace("–", "-").replace("-", "-"))
        # 匹配 D1-09:31:38 或 day1-09:31 等格式
        m = re.match(
            r"^(?:{dp}|d|day)\s*(\d+)[\-\s]?(\d{{1,2}}:\d{{2}}(?::\d{{2}})?)$".format(
                dp=re.escape(day_prefix)
            ),
            s,
            flags=re.IGNORECASE
        )
        if not m:
            return None
        day = int(m.group(1))  # 第几天
        hhmmss = m.group(2)  # HH:MM:SS
        parts = hhmmss.split(":")
        if len(parts) == 2:
            h, m1 = int(parts[0]), int(parts[1]); sec = 0
        else:
            h, m1, sec = int(parts[0]), int(parts[1]), int(parts[2])
        # 转换为绝对秒数：(day-1)*86400 + h*3600 + m*60 + s
        return (day - 1) * 86400 + h * 3600 + m1 * 60 + sec
    
    @classmethod
    def from_json(cls, timeline_json_path: str):
        """从timeline JSON文件加载"""
        with open(timeline_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        day_prefix = (data.get("day_prefix_used") or "D")
        segs = []
        for vname, meta in (data.get("mapping_by_input_label") or {}).items():
            st_lab = meta.get("start_label")  # 如 "D1-08:00:00"
            en_lab = meta.get("end_label")    # 如 "D1-12:00:00"
            st_abs = cls._parse_day_label(st_lab, day_prefix)
            en_abs = cls._parse_day_label(en_lab, day_prefix)
            if st_abs is None or en_abs is None:
                continue
            segs.append({
                "video": vname,
                "start_label": st_lab,
                "end_label": en_lab,
                "start_abs": st_abs,  # 绝对秒数（原始录制时间）
                "end_abs": en_abs,
                "merged_offset_start_seconds": float(meta.get("merged_offset_start_seconds", 0)),  # 合并视频中的起始秒数
                "merged_offset_end_seconds": float(meta.get("merged_offset_end_seconds", 0)),
                "description": meta.get("description", ""),
            })
        return cls(segs, day_prefix=day_prefix)

    def label_to_merged_seconds(self, label: Any) -> Optional[float]:
        """
        将时间标签转换为合并视频中的秒数
        输入可以是：
        - float/int: 直接返回
        - "HH:MM:SS": 转换为秒数
        - "D1-09:31:38": 查找对应片段，计算在合并视频中的位置
        """
        if label is None:
            return None
        if isinstance(label, (int, float)):
            return float(label)
        s = str(label).strip()
        # 尝试解析为 Dk-HH:MM:SS 格式
        abs_s = self._parse_day_label(s, self.day_prefix)
        if abs_s is not None:
            # 查找包含这个时间点的片段
            for seg in self.segments:
                if seg["start_abs"] <= abs_s <= seg["end_abs"]:
                    # 计算在合并视频中的位置
                    return seg["merged_offset_start_seconds"] + (abs_s - seg["start_abs"])
            return None  # 找不到对应片段
        # 尝试解析为 HH:MM:SS 格式
        try:
            if ":" in s:
                return parse_hhmmss(s)
        except Exception:
            pass
        return None

    def find_segment_for_offset(self, merged_sec: float) -> Optional[Dict[str, Any]]:
        for seg in self.segments:
            if seg["merged_offset_start_seconds"] <= merged_sec <= seg["merged_offset_end_seconds"]:
                return seg
        return None



def collect_questionstamps(qa_items: List[Dict[str, Any]], label_mapper: Optional[Any] = None) -> List[Tuple[float, List[Dict[str, Any]]]]:
    """
    根据 evidence.timestep.end 提取问题发生时间并排序。
    若提供 label_mapper（如 TimelineIndex.label_to_merged_seconds），
    会优先将类似 'D1-09:31:38' 的标签映射到合并视频的秒。
    返回 [(t_end_seconds, [item]), ...]
    """
    rows: List[Tuple[float, int, Dict[str, Any]]] = []  # (t_end, pos, item)

    for pos, it in enumerate(qa_items):
        ts = (it.get("evidence") or {}).get("timestep") or {}
        end_raw = ts.get("end") or ts.get("End") or ts.get("to")
        t_end = None
        # Timeline-based mapping first
        if label_mapper is not None:
            try:
                t_end = label_mapper(end_raw)
            except Exception:
                t_end = None
        # Fallbacks
        if t_end is None:
            try:
                if end_raw is None:
                    t_end = float("inf")
                elif isinstance(end_raw, (int, float)):
                    t_end = float(end_raw)
                else:
                    s = str(end_raw).strip()
                    t_end = parse_hhmmss(s) if ":" in s else float(s)
            except Exception:
                t_end = float("inf")
        rows.append((t_end, pos, it))

    rows.sort(key=lambda r: (r[0], r[1]))

    grouped: List[Tuple[float, List[Dict[str, Any]]]] = []
    for t_end, _, it in rows:
        grouped.append((t_end, [it]))
    return grouped






def get_audio_1s_at_time(video: "VideoFileClip", t_start: float, sr: int = 16000) -> np.ndarray:
    t0 = max(0.0, float(t_start))
    t1 = min(float(video.duration), t0 + 1.0)
    if getattr(video, "audio", None) is None or t1 <= t0:
        return np.zeros(int(sr), dtype=np.float32)
    try:
        snd = video.audio.subclip(t0, t1).to_soundarray(fps=sr)  # [N, C]
        if snd.ndim == 2:
            snd = snd.mean(axis=1)
        if len(snd) < sr:
            pad = np.zeros(sr - len(snd), dtype=snd.dtype)
            snd = np.concatenate([snd, pad], axis=0)
        snd = np.clip(np.nan_to_num(snd, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        return snd.astype(np.float32, copy=False)
    except Exception:
        return np.zeros(int(sr), dtype=np.float32)



_ASR = None
# Optional: Whisper ASR
USE_ASR = True
ASR_MODEL_NAME = "./weights/whisper-medium"
ASR_LANGUAGE   = "zh"
ASR_DEVICE     = 0 if torch.cuda.is_available() else -1

def init_asr():
    from transformers import pipeline
    global _ASR
    if _ASR is None:
        _ASR = pipeline(
            "automatic-speech-recognition",
            model=ASR_MODEL_NAME,
            device=ASR_DEVICE,
        )
    return _ASR


def transcribe_audio_1s(audio_np: Optional[np.ndarray], sr: int = 16000) -> str:
    if not USE_ASR or audio_np is None or len(audio_np) == 0:
        return ""
    try:
        asr = init_asr()
        audio_arr = np.asarray(audio_np, dtype=np.float32)
        audio_arr = np.clip(np.nan_to_num(audio_arr, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        out = asr(
            {"array": audio_arr, "sampling_rate": sr},
            return_timestamps=False,
            generate_kwargs={"task": "transcribe", "language": ASR_LANGUAGE},  # ★ 仅通过 generate_kwargs 指定
        )
        text = ""
        if isinstance(out, dict):
            text = (out.get("text") or "").strip()
        elif isinstance(out, list) and out and isinstance(out[0], dict):
            text = (out[0].get("text") or "").strip()
        if text:
            text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)
            if len(text) > 160:
                text = text[:160] + "…"
        # print(text)
        return text
    except Exception as e:
        print(f"[ASR] error: {e}")
        return ""


    
        


###############################################
# Prompt builders
###############################################

def build_question_prompt(item: Dict[str, Any]) -> str:
    qtype = (item.get("QA_type") or "").lower()
    question = item.get("question", "").strip()
    options: List[str] = item.get("options", [])

    if qtype == "mc_single":
        opt_str = "\n".join(options)
        instr = (
            "请根据提供的视频与音频内容回答一个单选题。\n"
            "只输出选项字母（例如 A），不要输出解释，"
        )
        return f"{instr}\n题目：{question}\n选项：\n{opt_str}\n你的答案是："

    if qtype == "mc_multi":
        opt_str = "\n".join(options)
        instr = (
            "请根据提供的视频与音频内容回答一个多选题。\n"
            "只输出所有正确选项字母，使用英文逗号分隔（例如 A,B），不要输出解释。"
        )
        return f"{instr}\n题目：{question}\n选项：\n{opt_str}\n你的答案是："

    if qtype == "binary":
        instr = (
            "请根据提供的视频与音频内容回答一个是非题。\n"
            "只输出一个单词：True 或 False，不要输出其他字符，不要输出解释。"
        )
        return f"{instr}\n题目：{question}\n你的答案是："

    instr = (
        "请根据提供的视频与音频内容简要作答本题。优先给出关键词序列或简短句子。\n"
        "在20个字以内为佳。"
    )
    return f"{instr}\n问题：{question}\n你的答案是："




###############################################
# Parsing and evaluation
###############################################
def parse_prediction(text: str, qtype: str):
    """解析模型输出为标准格式"""
    t = (text or "").strip()
    qtype = qtype.lower()
    
    if qtype == "mc_single":
        if re.search(r'[A-Z]\s*和\s*[A-Z]', t, re.IGNORECASE):
            # 检测 "A 和 B" 这种模式
            return []

        if re.search(r'[A-D]\s*,\s*[A-D]', t, re.IGNORECASE):
            # 检测 "A, B" 或 "A,B,C" 这种模式
            return []

        # 提取第一个大写字母
        letters = re.findall(r"[A-Z]", t.upper())
        return letters[:1] if letters else []
    
    if qtype == "mc_multi":
        # 提取所有大写字母并去重
        letters = re.findall(r"[A-Z]", t.upper())
        seen = set(); out = []
        for x in letters:
            if x not in seen:
                seen.add(x); out.append(x)
        return out
    
    if qtype == "binary":
        # 识别True/False及其变体
        truish = {"true", "t", "是", "对", "yes", "y", "正确"}
        falsish = {"false", "f", "否", "不对", "no", "n", "错误"}
        low = t.lower()
        if low in truish: return True
        if low in falsish: return False
        if re.search(r"\btrue\b", low): return True
        if re.search(r"\bfalse\b", low): return False
        return None
    
    # 开放式问题返回原文
    return t

def evaluate_item(gt: Dict[str, Any], pred) -> Dict[str, Any]:
    """
    评估单个答案
    返回：{"correct": bool, "metric": str, ...}
    """
    qtype = (gt.get("QA_type") or "").lower()
    ans = gt.get("answer", {}) or {}
    raw = ans.get("value", None)

    def _to_letters(x) -> list[str]:
        """将GT转换为字母列表"""
        if isinstance(x, list):
            s = ",".join(map(str, x))
        else:
            s = "" if x is None else str(x)
        return re.findall(r"[A-Z]", s.upper())

    def _to_bool(x) -> bool:
        """将GT转换为布尔值"""
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return bool(x)
        if isinstance(x, list) and x:
            return _to_bool(x[0])
        if isinstance(x, str):
            low = x.strip().lower()
            truish = {"true","t","1","y","yes","是","对","正确"}
            falsish = {"false","f","0","n","no","否","不对","错误"}
            if low in truish: return True
            if low in falsish: return False
        return False

    result = {"correct": False, "metric": ""}

    # 选择题
    if qtype in {"mc_single", "mc_multi"}:
        gt_letters = _to_letters(raw)
        if qtype == "mc_single":
            pred_letters = [str(x).upper() for x in (pred or [])]
            result["correct"] = (len(pred_letters) == 1 and pred_letters[0] in gt_letters)
            result["metric"] = "accuracy"
            return result
        # 多选题：集合完全相等
        pred_set = set([str(x).upper() for x in (pred or [])])
        gt_set = set(gt_letters)
        result["correct"] = (len(gt_set) > 0 and pred_set == gt_set)
        result["metric"] = "exact_set_match"
        return result

    # 判断题
    if qtype == "binary":
        gt_bool = _to_bool(raw)
        result["correct"] = (pred is not None and bool(pred) == gt_bool)
        result["metric"] = "accuracy"
        return result

    # 开放式问题：token overlap >= 25%
    if isinstance(raw, list) and raw:
        gt_text = str(raw[0])
    elif isinstance(raw, (str, int, float, bool)):
        gt_text = str(raw)
    else:
        gt_text = ""
    pred_text = (pred or "").strip().lower()
    gt_text_l = gt_text.strip().lower()

    def tokens(s: str):
        return [w for w in re.findall(r"[一-龥A-Za-z0-9]+", s) if w]

    gts = tokens(gt_text_l)
    prs = set(tokens(pred_text))
    overlap = [w for w in gts if w in prs]
    result["correct"] = len(overlap) >= max(1, len(gts) // 4)
    result["metric"] = "token_overlap>=25%"
    result["overlap_tokens"] = overlap
    return result

###############################################
# Seeding
###############################################
def set_global_seed(seed: int = 42):
    """设置全局随机种子，保证可复现"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import random as _random
        _random.seed(seed)
    except Exception:
        pass



###############################################
# LLM judge (开放式问题评分)
###############################################
prompt_dict = {
    "open_ended_cn_v1": (
        "你是一个客观判分器。请只输出一个 JSON 对象（严格），不要包含多余文本或解释。\n"
        "给定：\n"
        "【问题】'{question}\n"
        "【参考答案】'{ground_truth}\n"
        "【模型回答】'{prediction}\n\n"
        "请根据模型回答与参考答案在事实与要点上的一致程度打分，分值为 1～5 的整数：\n"
        "5 = 完全正确；4 = 基本正确；3 = 部分正确；2 = 大多不正确；1 = 完全错误。\n\n"
        "只输出以下JSON（严格）：例如 {{\"score\": 4}}\n"
    ),
}

def evaluate_with_llm(question: str, ground_truth: str, prediction: str, llm_config: Dict = None) -> Dict[str, any]:
    """使用GPT-4o对开放式问题评分（1-5分）"""
    import json as _json
    import traceback
    
    # 默认配置
    default_llm_config = {
        "enabled": True,
        "provider": "azure",
        "api_key": API_KEY,
        "azure_endpoint": END_POINT,
        "azure_api_version": API_VERSION,
        "azure_deployment": ENGINE,
        "temperature": 0.0,
        "timeout": 30,
        "prompt": "open_ended_cn_v1",
    }
    if llm_config is None:
        llm_config = {}
    for k, v in default_llm_config.items():
        llm_config.setdefault(k, v)

    if not llm_config.get("enabled", False):
        return {"llm_score": 0}

    try:
        import openai
        try:
            from openai import AzureOpenAI
        except Exception:
            AzureOpenAI = None

        prompt_name = llm_config["prompt"]
        assert prompt_name in prompt_dict
        prompt = prompt_dict[prompt_name].format(
            question=str(question or ""),
            ground_truth=str(ground_truth or ""),
            prediction=str(prediction or "")
        )

        provider = str(llm_config.get("provider", "azure")).lower()

        if provider == "azure":
            required = ["api_key", "azure_endpoint", "azure_api_version", "azure_deployment"]
            missing = [k for k in required if not llm_config.get(k)]
            if missing:
                return {"llm_score": 0}
            if AzureOpenAI is None:
                return {"llm_score": 0}

            client = AzureOpenAI(
                api_key=llm_config["api_key"],
                api_version=llm_config["azure_api_version"],
                azure_endpoint=llm_config["azure_endpoint"],
            )

            response = client.chat.completions.create(
                model=llm_config["azure_deployment"],
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=llm_config.get("temperature", 0.0),
                timeout=llm_config.get("timeout", 30),
            )
        else:
            client = openai.OpenAI(api_key=llm_config.get("api_key", "dummy"))
            response = client.chat.completions.create(
                model=llm_config.get("model", "gpt-4o"),
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=llm_config.get("temperature", 0.0),
                timeout=llm_config.get("timeout", 30),
            )

        content = response.choices[0].message.content
        data = _json.loads(content)
        score_raw = data.get("score", 0)
        try:
            score = int(score_raw)
        except Exception:
            try:
                score = int(str(score_raw).strip())
            except Exception:
                score = 0
        if 1 <= score <= 5:
            return {"llm_score": score}
        else:
            return {"llm_score": 0}
    except AssertionError:
        raise
    except Exception:
        print("⚠️ LLM evaluation error:", traceback.format_exc())
        return {"llm_score": 0}




def build_single_unit(video: "VideoFileClip", current_sec: int, ret_video: bool = False, asr_txt: bool = False, sr: int = 16000):
    """
    构建视频的秒单元：每秒提取1帧图像 + 1秒音频
    返回：(units列表, 视频时长, 音频采样率)
    """
    
    t_start = float(current_sec)
    t_end = min(current_sec + 1, float(video.duration) - 1e-3)

    if t_end > t_start:
        t_frame = random.uniform(t_start, t_end)
    else:
        t_frame = t_start

    frame = video.get_frame(t_frame)
    image = Image.fromarray(frame.astype(np.uint8))

    if hasattr(video, "subclipped"):
        video = video.subclipped(t_start, t_end)
    elif hasattr(video, "subclip"):
        video = video.subclip(t_start, t_end)

    if ret_video:
        return video

    # 提取音频为16kHz单声道
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
        wav_path = tf.name
        video.audio.write_audiofile(wav_path, codec="pcm_s16le", fps=16000, logger=None)
        audio_np, sr = librosa.load(wav_path, sr=16000, mono=True)
      
    if asr_txt:
        text = transcribe_audio_1s(audio_np, sr=sr)
        return (image, text)
    else:
        return (image, audio_np)



def pil_to_data_uri(image: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def encode_audio_np_to_b64_wav(audio_np, sr=16000) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio_np, sr, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()
    return base64.b64encode(wav_bytes).decode("utf-8")


def video_clip_to_mp4_bytes(video_clip: "VideoFileClip") -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        temp_path = tf.name

    video_clip.write_videofile(
        temp_path,
        codec="libx264",
        audio_codec="aac",
        fps=1,
        logger=None,
    )

    with open(temp_path, "rb") as f:
        video_bytes = f.read()

    os.remove(temp_path)
    return video_bytes



class GPUMonitor:
    def __init__(self, device_id=0, save_dir="./"):
        self.device_id = device_id
        self.save_dir = Path(save_dir)
        self.gpu_name = torch.cuda.get_device_name(device_id)
        self.total_gb = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)  # 改为GB
        
        self.times = []
        self.allocated = []
        self.start_time = time.time()
        
        # 实时保存的文件路径
        self.realtime_log = self.save_dir / "gpu_memory_realtime.jsonl"
        
        print(f"GPU监控启动: {self.gpu_name} ({self.total_gb:.1f}GB)")
    
    def _save_record(self, t, mem_gb, label):
        """实时保存单条记录到文件（追加模式）"""
        record = {
            "time": f"{t:.2f}",
            "label": label,
            "mem_gb": round(mem_gb, 3),
            "utilization": round(mem_gb/self.total_gb*100, 2)
        }
        
        # 追加写入到jsonl文件
        with open(self.realtime_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    def record(self, label=""):
        """记录当前显存"""
        t = time.time() - self.start_time
        free_mem, total_mem = torch.cuda.mem_get_info(self.device_id)
        used_gb = (total_mem - free_mem) / (1024**3)

        self.times.append(t)
        self.allocated.append(used_gb)
        
        if label:
            print(f"[{t:6.1f}s] {label:20s} | {used_gb:6.2f}GB ({used_gb/self.total_gb*100:5.1f}%)")
        
        # 实时保存到文件
        self._save_record(t, used_gb, label)
    
    def plot(self, save_path=None):
        """绘制显存曲线"""
        if not self.times:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 显存使用（GB）
        ax1.plot(self.times, self.allocated, 'b-', linewidth=2)
        ax1.fill_between(self.times, 0, self.allocated, alpha=0.3)
        ax1.axhline(self.total_gb, color='r', linestyle='--', label=f'总显存 {self.total_gb:.1f}GB')
        ax1.set_ylabel('显存 (GB)')
        ax1.set_title(f'GPU显存监控 - {self.gpu_name}')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 利用率
        util = [m/self.total_gb*100 for m in self.allocated]
        ax2.plot(self.times, util, 'g-', linewidth=2)
        ax2.fill_between(self.times, 0, util, alpha=0.3, color='g')
        ax2.axhline(90, color='orange', linestyle='--', alpha=0.5)
        ax2.set_xlabel('时间 (秒)')
        ax2.set_ylabel('利用率 (%)')
        ax2.set_ylim(0, 105)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.save_dir / "gpu_memory.png"
        plt.savefig(save_path, dpi=120)
        plt.close()
        print(f"✓ 显存图表: {save_path}")
    
    def save(self, save_path=None):
        """保存汇总数据"""
        if save_path is None:
            save_path = self.save_dir / "gpu_memory_summary.json"
        
        data = {
            "gpu": self.gpu_name,
            "total_gb": round(self.total_gb, 2),
            "peak_gb": round(max(self.allocated), 3) if self.allocated else 0,
            "avg_gb": round(sum(self.allocated)/len(self.allocated), 3) if self.allocated else 0,
            "num_records": len(self.allocated),
            "realtime_log": str(self.realtime_log)
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ 显存汇总: {save_path}")
        print(f"  峰值: {data['peak_gb']:.2f}GB, 平均: {data['avg_gb']:.2f}GB")
        print(f"✓ 实时记录: {self.realtime_log} (共{data['num_records']}条)")



def log_token_usage(response, log_file="token_usage.jsonl"):
    """记录每次 API 调用的 token 使用情况"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")



def log_token_usage_gemini(usage_metadata, current_sec, log_file="token_usage.jsonl"):
    """记录 Gemini API 调用的 token 使用情况"""
    try:
        # 提取模态细节
        modality_details = {}
        if hasattr(usage_metadata, 'prompt_tokens_details'):
            for modal in usage_metadata.prompt_tokens_details:
                modality_details[modal.modality.name] = modal.token_count
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "current_sec": current_sec,
            "prompt_tokens": usage_metadata.prompt_token_count,
            "candidates_tokens": usage_metadata.candidates_token_count,
            "total_tokens": usage_metadata.total_token_count,
            "thoughts_tokens": getattr(usage_metadata, 'thoughts_token_count', 0),
            "output_tokens": getattr(usage_metadata, 'thoughts_token_count', 0) + usage_metadata.candidates_token_count,
            "modality_details": modality_details,  # VIDEO, AUDIO, TEXT分类
        }
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        
        return log_entry
    except Exception as e:
        print(f"[WARNING] 无法记录token使用: {e}")
        return None

