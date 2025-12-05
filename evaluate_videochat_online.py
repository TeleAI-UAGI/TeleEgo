import types
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
import utils
from utils import ProgressLogger, TimelineIndex
import warnings
warnings.filterwarnings('ignore')
from internvl.model.videochat_online import (
    VideoChatOnline_IT,
    InternVLChatConfig,
)
from internvl.train.dataset import build_transform, dynamic_preprocess
from transformers import StoppingCriteria, StoppingCriteriaList
from utils import TimeLimitStoppingCriteria
from collections import deque
import argparse
import os
from datetime import datetime

# from internvl.model.internvl_chat import InternVLChatModel_IT
from internvl.train.constants import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
    VIDEO_CONTEXT_TOKEN,
)



def stream_with_timeout(image, question):
    
    stopping_criteria = StoppingCriteriaList([
        TimeLimitStoppingCriteria(max_time_seconds=args.decision_window)
    ])

    gen_cfg = dict(
        max_new_tokens=args.max_new_tokens,
    )
    gen_cfg["stopping_criteria"] = stopping_criteria

    if args.use_history and history is not None:
        history_list = list(history)
        if history_list:
            last_q, last_a = history_list[-1]
            last_q = last_q.replace("<image>", "")   # 去掉最后一个 <image> （前面的<image>已经都去掉了）
            history_list[-1] = (last_q, last_a)
    else:
        history_list = None

    start_time = time.time()

    pred = model.chat(
                        tokenizer,
                        image,
                        question,
                        gen_cfg,
                        num_patches_list=[1],
                        verbose=False,
                        history=history_list,
                        return_history=args.use_history
                    )

    if args.use_history:
        pred, new_history_list = pred
        history.clear()
        history.extend(new_history_list)

    elapsed_time = time.time() - start_time
    
    if not args.use_history:
        assert history is None

    return pred, elapsed_time




def evaluate_single():
    """
    单个视频-QA对的完整评估流程
    
    流程：
    1. 加载QA数据和视频units
    2. 构建问题队列（包含initial和recall）
    3. 逐秒输入units，到达问题时间点时提问
    4. 5秒内必须回答完，否则判错
    5. 答对的问题会在60s、120s、...后再问（recall），测试记忆持久性
    6. 计算RTA和MPT指标
    """
    
    # ========== 1. 加载数据 ==========
    with open(qa_path, "r", encoding="utf-8") as f:
        qa_items = json.load(f)
    
    timeline_idx = TimelineIndex.from_json(timeline_path) # 加载timeline（用于时间戳映射）
    
    video = VideoFileClip(video_path, fps_source="fps")
    duration = float(video.duration)
    n_units = int(math.ceil(video.duration))
    
    # ========== 2. 提取时间戳并构建初始队列 ==========
    def extract_timestamp(item):
        """从item中提取问题时间戳（evidence.timestep.end）"""

        ts = (item.get("evidence") or {}).get("timestep") or {}
        end_raw = ts.get("end") or ts.get("End") or ts.get("to")
        t_end = timeline_idx.label_to_merged_seconds(end_raw)
        return t_end
    
    # 构建问题队列：[(t_end, item, phase, round_id), ...]; phase: "initial" 或 "recall"; round_id: 0表示initial，1-10表示recall轮次
    qa_queue = []
    for item in qa_items:
        t_end = extract_timestamp(item)
        if t_end is not None and math.isfinite(t_end):
            qa_queue.append((t_end, item, "initial", 0))
    
    qa_queue.sort(key=lambda x: (x[0], x[2] == "recall", x[3])) # 按时间排序（相同时间initial优先，相同phase按round排序）
    
    os.makedirs(save_dir, exist_ok=True)
    results = []  # 存储所有问题的结果
    
    # ========== MPT追踪 ==========
    mpt_tracker = {}  # mpt_tracker: {item_index: (t_star, [round_results])}  t_star: 首次答对的时间；round_results: [initial结果, recall轮1结果, recall轮2结果, ...]
    
    # ========== 3. 流式输入主循环 ==========
    current_sec = 0  # 当前已输入到第几秒
    qa_idx = 0  # 队列中下一个要处理的问题索引

    announced_videos = set()  # for segment intro messages (insert once per child video)
    
    print(f"开始流式输入，共 {n_units} 秒，{len(qa_queue)} 个问题（不含recall）...")
    with tqdm.tqdm(total=n_units+600, desc="流式输入") as pbar:
        
        # 主循环：继续直到视频结束且队列清空
        while current_sec < n_units or qa_idx < len(qa_queue):
            current_sec = pbar.n
            
            # ========== 3.1 检查当前时间点的问题 ==========
            questions_at_current = []  # 收集所有在 current_sec+1 时刻需要提问的问题
            while qa_idx < len(qa_queue):
                t_end, item, phase, round_id = qa_queue[qa_idx]
                question_sec = int(math.ceil(t_end))  # 问题应该在哪一秒提问
                
                if question_sec <= current_sec + 1:
                    # 需要在当前或之前提问
                    questions_at_current.append((t_end, item, phase, round_id))
                    qa_idx += 1
                else:
                    break # 后面的问题时间更晚，暂不处理
            
            # 视频还在播放,取当前帧，否则取最后一帧
            video_sec = current_sec if current_sec <= n_units - 1 else n_units - 1
            seg = timeline_idx.find_segment_for_offset(video_sec)
            if seg and seg["video"] not in announced_videos:
                announced_videos.add(seg["video"])
                model.system_message = (
                    f"【当前子视频段信息】人物/场景描述：{seg.get('description','') or '（无）'}。"
                )

            image, audio = utils.build_single_unit(video, video_sec, asr_txt=True)
            image = image_transform(image)
            image = image.unsqueeze(0).to(torch.bfloat16).cuda()
            text_prompt = (
                    f"<image> \n这是第{video_sec+1}帧（第{video_sec+1}秒）的画面,\n"
                    f"此时的音频转写是：{audio}。\n"
            )

            # ========== 3.2 如果没有问题，继续输入下一秒 ==========
            if not questions_at_current:
                instruction = (
                    f"请仔细观察并记住这一帧的内容，20个字以内为佳。\n"
                    f"你的回答是："
                )
                prompt = text_prompt + instruction
                pred, elapsed_time = stream_with_timeout(image, prompt)

            # ========== 3.3 如果有问题，处理当前时间点的所有问题 ==========
            else:
                for t_end, item, phase, round_id in questions_at_current:

                    question_sec = int(math.ceil(t_end))
                    assert current_sec + 1 == question_sec
                    
                    # ========== 3.3.1 构建问题提示词 ==========
                    qtype = (item.get("QA_type") or "").lower()
                    question = utils.build_question_prompt(item)
                    prompt = text_prompt + question
                    pred, elapsed_time = stream_with_timeout(image, prompt)

                    # ========== 3.3.2 解析和评估答案 ==========
                    parsed = utils.parse_prediction(pred, qtype)
                    eval_res = utils.evaluate_item(item, parsed)
                    
                    # ========== 3.3.3 LLM评分（仅开放式问题） ==========
                    llm_score = 0
                    if qtype not in {"mc_single", "mc_multi", "binary"} and llm_eval_config.get("enabled"):
                        assert qtype == "open_ended"

                        gt_vals = (item.get("answer", {}) or {}).get("value", [])
                        ground_truth = "" if not gt_vals else str(gt_vals[0])
                        llm_out = utils.evaluate_with_llm(
                            question=item.get("question", ""),
                            ground_truth=ground_truth,
                            prediction=pred,
                            llm_config=llm_eval_config
                        )
                        llm_score = int(llm_out.get("llm_score", 0))
                    
                    # ========== 3.3.4 记录结果 ==========
                    rec = {
                        "phase": phase,
                        "round": round_id if phase == "recall" else None,
                        "index": item.get("index"),
                        "category": item.get("category"),
                        "subcategory": item.get("subcategory"),
                        "QA_type": qtype,
                        "question": item.get("question"),
                        "options": item.get("options", []),
                        "gold": item.get("answer", {}).get("value", []),
                        "questionstamp": t_end,
                        "pred_text": pred,
                        "parsed_pred": parsed,
                        "correct": eval_res.get("correct", False),
                        "metric": eval_res.get("metric", ""),
                        "elapsed_time": elapsed_time,
                        "llm_score": llm_score if qtype not in {"mc_single","mc_multi","binary"} else None,
                    }

                    if phase == "recall":
                        rec["recall_delay"] = round_id * args.recall_delay
                    if "overlap_tokens" in eval_res:
                        rec["overlap_tokens"] = eval_res["overlap_tokens"]

                    results.append(rec)

                     # realtime_results, 直接保存JSON   
                    idx = item.get("index", "unknown")
                    if phase == "initial":
                        filename = f"q{idx}_initial.json"
                    else:
                        filename = f"q{idx}_recall_r{round_id}.json"

                    json_path = Path(realtime_results_folder) / filename
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(rec, f, ensure_ascii=False, indent=2)
                    
                    # ========== 3.3.5 统计（仅initial阶段） ==========
                    if phase == "initial":
                        # ========== 3.3.5.1 如果答对，安排recall ==========
                        if rec["correct"]:
                            idx = item.get("index")
                            mpt_tracker[idx] = (t_end, [True])  # 记录t_star和初始结果
                            recall_entries = []  # 收集所有recall问题

                            for r in range(1, args.max_recall_rounds + 1):
                                recall_t = t_end + r * args.recall_delay
                                new_entry = (recall_t, item, "recall", r)
                                recall_entries.append(new_entry)

                            qa_queue.extend(recall_entries)
                            qa_queue[qa_idx:] = sorted(
                                qa_queue[qa_idx:],
                                key=lambda x: (x[0], x[2] == "recall", x[1].get("index"))
                            )  # 排序

                    # ========== 3.3.6 MPT追踪：recall结果 ==========
                    if phase == "recall":
                        idx = item.get("index")
                        assert idx in mpt_tracker, f"Recall问题但index {idx} 不在tracker中"
                        mpt_tracker[idx][1].append(rec["correct"])

                        if not rec["correct"]:  # 如果失败，移除该item的后续recall，从后往前删除
                            i = len(qa_queue) - 1
                            while i >= qa_idx:
                                t, it, ph, rd = qa_queue[i]
                                if it.get("index") == idx and ph == "recall" and rd > round_id:
                                    qa_queue.pop(i)
                                i -= 1

                    # ========== 3.3.7 打印日志 ==========
                    if qtype in {"mc_single", "mc_multi", "binary"}:
                        print(f"[t={t_end:.2f}s][#{item.get('index')}][{phase}] {qtype} | correct={rec['correct']} | {elapsed_time:.2f}s")
                    else:
                        score_str = "N/A" if llm_score is None else str(llm_score)
                        print(f"[t={t_end:.2f}s][#{item.get('index')}][{phase}] {qtype} | score={score_str} | {elapsed_time:.2f}s")
            
            pbar.update(1)
            continue

    
    # ========== 保存结果 ==========
    out_path = Path(save_dir) / "eval_predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    mpt_out_path = Path(save_dir) / "mpt_results.json"
    with open(mpt_out_path, "w", encoding="utf-8") as f:
        json.dump(mpt_tracker, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 结果保存到: {out_path} 和 {mpt_out_path}")
    
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--base_dir", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--p_num", type=str, default="1")
    parser.add_argument("--qa_suffix", type=str, default="A")

    parser.add_argument("--decision_window", type=float, default=5.0)
    parser.add_argument("--recall_delay", type=float, default=60.0)
    parser.add_argument("--max_recall_rounds", type=int, default=10)

    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--api_version", type=str, default="")
    parser.add_argument("--end_point", type=str, default="")
    parser.add_argument("--engine", type=str, default="")

    parser.add_argument("--use_history", action="store_true")
    parser.add_argument("--history_max_len", type=int, default=300)

    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args()

    utils.set_global_seed(args.seed)

    video_path = f"{args.base_dir}/video_merged/merged_P{args.p_num}.mp4"
    qa_path = f"{args.base_dir}/QAs/merged_P{args.p_num}_{args.qa_suffix}.json"
    timeline_path = f"{args.base_dir}/video_merged/timeline_P{args.p_num}.json"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_pred_path = f"{args.base_dir}/outputs/videochat-online/p{args.p_num}/{timestamp}/eval_predictions_P{args.p_num}.json"
    
    save_dir = str(Path(save_pred_path).parent)
    os.makedirs(save_dir, exist_ok=True)

    config_path = Path(save_dir) / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    realtime_results_folder = Path(save_dir) / f"realtime_results"
    os.makedirs(realtime_results_folder, exist_ok=True)

    llm_eval_config = {
        "enabled": True,
        "provider": "azure",
        "api_key": args.api_key,
        "azure_endpoint": args.end_point,
        "azure_api_version": args.api_version,
        "azure_deployment": args.engine,
        "temperature": 0.0,
        "timeout": 30,
        "prompt": "open_ended_cn_v1",
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        add_eos_token=False,
        trust_remote_code=True,
        use_fast=False,
    )

    # tokenizer.model_max_length = 8192 * 4
    tokenizer.init_kwargs["truncation"] = True
    tokenizer.truncation = True
    tokenizer.truncation_side = "left"
    tokenizer.init_kwargs["truncation_side"] = "left"
    
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    config = InternVLChatConfig.from_pretrained(
        args.checkpoint, trust_remote_code=True
    )
    model = (
        VideoChatOnline_IT.from_pretrained(
            args.checkpoint,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )

    model.img_context_token_id = img_context_token_id
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    image_transform = build_transform(is_train=False, input_size=448)

    history = deque(maxlen=args.history_max_len) if args.use_history else None

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20:
        NUM_BEAMS = 1
        print(f"[test] total_params: {total_params}B, use num_beams: {NUM_BEAMS}")
    else:
        print(f"[test] total_params: {total_params}B")
    
    print(f"[test] image_size: {image_size}")
    print(f"[test] template: {model.config.template}")
    
    # 运行评估
    evaluate_single()
    
