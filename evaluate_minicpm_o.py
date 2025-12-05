import types, copy
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
from transformers import StoppingCriteria, StoppingCriteriaList
from utils import ProgressLogger, TimelineIndex, TimeLimitStoppingCriteria
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

import logging, os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.disable(logging.WARNING)


BAD_ENDINGS = ('</div>', '<div>', '</p>', '<p>', '\\n', '\n')




def stream_with_timeout(session_id):

    stopping_criteria = StoppingCriteriaList([
        TimeLimitStoppingCriteria(max_time_seconds=args.decision_window)
    ])

    pred = ""

    start_time = time.time()

    res = model.streaming_generate(
        session_id=session_id,
        tokenizer=tokenizer,
        generate_audio=False,
        max_new_tokens=args.max_new_tokens,
        stopping_criteria=stopping_criteria
    )
    
    for chunk in res:
        pred += chunk.get("text")

    elapsed_time = time.time() - start_time
    
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
    mpt_tracker = {} # mpt_tracker: {item_index: (t_star, [round_results])}  t_star: 首次答对的时间；round_results: [initial结果, recall轮1结果, recall轮2结果, ...]
    
    # ========== 4. 流式输入主循环 ==========
    current_sec = 0  # 当前已输入到第几秒
    qa_idx = 0  # 队列中下一个要处理的问题索引

    announced_videos = set()

    model.reset_session()
    session_id = f"streaming_session"
    
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
            
            video_sec = current_sec if current_sec <= n_units - 1 else n_units - 1  # 视频还在播放,取当前帧，否则取最后一帧
            seg = timeline_idx.find_segment_for_offset(video_sec)
            if seg and seg["video"] not in announced_videos:
                announced_videos.add(seg["video"])
                sys_text = (
                    f"【当前子视频段信息】人物/场景描述：{seg.get('description','') or '（无）'}。"
                )

            # 10min新开一个会话
            if current_sec % 600 == 0:  
                model.reset_session()
                session_id = f"streaming_session_{current_sec}s"
    
            image, audio = utils.build_single_unit(video, video_sec)
            unit_content = [sys_text, "<unit>", image, audio]
            
            # ========== 3.2 如果没有问题，继续输入下一秒 ==========
            if not questions_at_current:
                _ = model.streaming_prefill(
                    session_id=session_id,
                    msgs=[{"role": "user", "content": unit_content}],
                    tokenizer=tokenizer
                )
            
            # ========== 3.3 如果有问题，处理当前时间点的所有问题 ==========
            else:
                for t_end, item, phase, round_id in questions_at_current:
                    
                    question_sec = int(math.ceil(t_end))
                    assert current_sec + 1 == question_sec

                    # ========== 构建问题提示词 ==========
                    qtype = (item.get("QA_type") or "").lower()
                    question = utils.build_question_prompt(item)

                    unit_input = copy.deepcopy(unit_content)
                    unit_input.append(question)
                    
                    # Prefill问题
                    _ = model.streaming_prefill(
                        session_id=session_id,
                        msgs=[{"role": "user", "content": unit_input}],
                        tokenizer=tokenizer
                    )
                    
                    # ========== 流式生成（5秒超时） ==========
                    cache_len = model.llm_past_key_values[0][0].shape[2]
                    print(f"[t={current_sec}s] KV Cache长度: {cache_len}")
                    pred, elapsed_time = stream_with_timeout(session_id)
                    
                    # ========== 解析和评估答案 ==========
                    parsed = utils.parse_prediction(pred, qtype)
                    eval_res = utils.evaluate_item(item, parsed)
                    
                    # ========== LLM评分（仅开放式问题） ==========
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
                    
                    # ========== 记录结果 ==========
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
                        "cache_len": cache_len
                    }

                    if phase == "recall":
                        rec["recall_delay"] = round_id * args.recall_delay
                    if "overlap_tokens" in eval_res:
                        rec["overlap_tokens"] = eval_res["overlap_tokens"]

                    results.append(rec)

                     # 直接保存JSON
                    idx = item.get("index", "unknown")
                    if phase == "initial":
                        filename = f"q{idx}_initial.json"
                    else:
                        filename = f"q{idx}_recall_r{round_id}.json"

                    json_path = Path(realtime_results_folder) / filename
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(rec, f, ensure_ascii=False, indent=2)

                    # ========== 统计（仅initial阶段） ==========
                    if phase == "initial":
                        
                        # ========== 如果答对，安排recall ==========
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

                    # ========== MPT追踪：recall结果 ==========
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
                    
                    # ========== 打印日志 ==========
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
    save_pred_path = f"{args.base_dir}/outputs/minicpm_o/p{args.p_num}/{timestamp}/eval_predictions_P{args.p_num}.json"

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

    model = AutoModel.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)

    model.terminators.append('<|tts_eos|>')
   
    # 获取系统提示
    sys_msg = model.get_sys_prompt(mode='omni', language='zh')
    
    # 运行评估
    evaluate_single()
    