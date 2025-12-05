#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
from collections import defaultdict
import pandas as pd


# ========== 配置区域 ==========

BASE_DIR = "/gemini/space/zxl/TeleEgo/teleego_data/outputs"
QA_DIR = "/gemini/space/zxl/TeleEgo/teleego_data/QAs"

RECALL_INTERVAL = 60

# 手动配置每个模型每个视频的时间戳
# 格式: {model_name: {video_id: timestamp}}
MODEL_TIMESTAMPS = {
    "videochat-online": {
        "p1": "",  # 填入实际的时间戳, 如 20251119_19390
        "p2": "",
        "p3": "",
        "p4": "",
        "p5": "",
    },
    
    "qwen25_vl": {
        "p1": "",  
        "p2": "",
        "p3": "",
        "p4": "",
        "p5": "",
    },

    "qwen25_omni": {
        "p1": "",
        "p2": "",
        "p3": "",
        "p4": "",
        "p5": "",
    },
    

    "minicpm_o": {
        "p1": "",
        "p2": "",
        "p3": "",
        "p4": "",
        "p5": "",
    },
    
    "gpt-4o": {
        "p1": "",
        "p2": "",
        "p3": "",
        "p4": "",
        "p5": "",
    },
    
    "gemini25_pro": {
        "p1": "",
        "p2": "",
        "p3": "",
        "p4": "",
        "p5": "",
    },
}


# 类别映射（根据 TeleEgo benchmark）
CATEGORY_MAPPING = {
    "Memory": [
        "Ultra-long Memory",
        "Short-term Memory", 
        "Entity Tracking",
        "Temporal Comparison & Interval",
        "Long-term Memory"
    ],
    "Understanding": [
        "Intent Inference",
        "Causal Understanding",
        "Cross-modal Understanding",
        "Multi-step Reasoning"
    ],
    "Cross-Memory Reasoning": [
        "Cross-entity Relation",
        "Temporal Chain Understanding",
        "Cross-temporal Causality"
    ]
}

# 小类缩写映射（用于表格显示）
SUBCATEGORY_ABBR = {
    "Ultra-long Memory": "UlM",
    "Short-term Memory": "StM",
    "Entity Tracking": "ET",
    "Temporal Comparison & Interval": "TCI",
    "Long-term Memory": "LtM",
    "Intent Inference": "II",
    "Causal Understanding": "CU",
    "Cross-modal Understanding": "CmU",
    "Multi-step Reasoning": "MsR",
    "Cross-entity Relation": "CeR",
    "Temporal Chain Understanding": "TCU",
    "Cross-temporal Causality": "CtC"
}


def load_json(json_path):
    """加载预测结果 JSON 文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_qa_info(video_id):
    """加载 QA 信息（用于获取 category 和 subcategory）"""
    # 根据视频 ID 构建 QA 文件路径
    # 假设格式为 merged_P{video_id}_A.json
    qa_file = os.path.join(QA_DIR, f"merged_{video_id.upper()}_A.json")
    
    if not os.path.exists(qa_file):
        print(f" 未找到 QA 文件: {qa_file}")
        return {}
    
    qa_data = load_json(qa_file)
    
    # 构建 index -> {category, subcategory, QA_type} 的映射
    qa_info = {}
    for item in qa_data:
        idx = item.get("index")
        if idx:
            qa_info[idx] = {
                "category": item.get("category"),
                "subcategory": item.get("subcategory"),
                "QA_type": item.get("QA_type")
            }
    
    return qa_info


def calculate_rta_for_model(model_name, json_files):
    """
    计算单个模型的 RTA 指标
    """
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for json_file in json_files:
        # print(f"  处理: {os.path.basename(json_file)}")
        predictions = load_json(json_file)
        
        for item in predictions:
            if item.get("phase") != "initial" or item.get("QA_type") == "open_ended":
                continue
            
            subcategory = item.get("subcategory")
            correct = item.get("correct", False)
            
            if subcategory:
                stats[subcategory]["total"] += 1
                if correct:
                    stats[subcategory]["correct"] += 1
    
    results = {}
    
    # 计算每个小类的准确率
    for subcategory in stats:
        total = stats[subcategory]["total"]
        correct = stats[subcategory]["correct"]
        accuracy = (correct / total * 100) if total > 0 else 0.0
        results[subcategory] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    
    # 计算每个大类的准确率
    for category, subcategories in CATEGORY_MAPPING.items():
        total_correct = 0
        total_count = 0
        
        for subcat in subcategories:
            if subcat in stats:
                total_correct += stats[subcat]["correct"]
                total_count += stats[subcat]["total"]
        
        accuracy = (total_correct / total_count * 100) if total_count > 0 else 0.0
        results[category] = {
            "accuracy": accuracy,
            "correct": total_correct,
            "total": total_count
        }
    
    # 计算 Overall 准确率
    overall_correct = sum(s["correct"] for s in stats.values())
    overall_total = sum(s["total"] for s in stats.values())
    overall_accuracy = (overall_correct / overall_total * 100) if overall_total > 0 else 0.0
    
    results["Overall"] = {
        "accuracy": overall_accuracy,
        "correct": overall_correct,
        "total": overall_total
    }

    rta_json_path = os.path.join(BASE_DIR, f"rta_{model_name}.json")
    with open(rta_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f" RTA 结果已保存: {rta_json_path}")
    
    return results



def calculate_mpt_for_model(model_name, video_timestamps):
    """
    计算单个模型的 MPT 指标
    
    MPT = 总记忆时间 / 问题数
    
    规则:
    - 第一次回答错误的问题: MPT = 0
    - 第一次回答正确但后续失败的问题: MPT = 失败时的累计时间
    - 始终正确的问题: MPT = 最大时间（10轮 × 60s = 600s）
    - 不在 mpt_results.json 但在 eval_predictions.json 里的: MPT = 0
    - 不在 eval_predictions.json 里的: 不计入总数
    """
    
    stats = defaultdict(lambda: {"total_time": 0.0, "count": 0})
    
    for vid, timestamp in video_timestamps.items():
        if not timestamp:
            continue
        
        # 加载 eval_predictions.json
        eval_file = os.path.join(BASE_DIR, model_name, vid, timestamp, "eval_predictions.json")
        if not os.path.exists(eval_file):
            continue
        
        predictions = load_json(eval_file)
        
        # 加载 mpt_results.json
        mpt_file = os.path.join(BASE_DIR, model_name, vid, timestamp, "mpt_results.json")
        mpt_data = load_json(mpt_file)
        
        # 加载 QA 信息
        qa_info = load_qa_info(vid)
        
        print(f"处理 MPT: {vid} - {timestamp}")
        
        # 构建 index -> initial correct 的映射
        initial_results = {}
        for item in predictions:
            if item.get("phase") == "initial":
                idx = item.get("index")
                qa_type = item.get("QA_type")
                subcategory = item.get("subcategory")
                correct = item.get("correct", False)
                
                # 忽略 open_ended
                if qa_type == "open_ended":
                    continue
                
                initial_results[idx] = {
                    "correct": correct,
                    "subcategory": subcategory,
                    "qa_type": qa_type
                }
        
        # 处理每个问题的 MPT
        for idx_str, value in mpt_data.items():
            idx = int(idx_str)

            # 检查该问题是否在 initial_results 中
            if idx not in initial_results:
                continue
            
            # 获取问题信息
            subcategory = initial_results[idx]["subcategory"]
            
            # 解析 mpt_data 的值
            # 格式: [timestamp, [true, true, false, ...]]
            t_star, recall_results = value
            
            # 计算 MPT
            if initial_results[idx]["correct"]:
                # 第一次回答正确，计算失败前的累计时间
                mpt = 0
                for i, result in enumerate(recall_results):
                    if result:
                        # 回答正确，继续累计时间
                        mpt += RECALL_INTERVAL
                    else:
                        # 回答错误，停止累计
                        break
                
                # 如果全部正确，mpt 已经累计了所有时间
                # 注意: recall_results 长度最多为 10
            else:
                # 第一次回答错误
                assert 0
            
            stats[subcategory]["total_time"] += mpt
            stats[subcategory]["count"] += 1
        
        # 处理不在 mpt_results.json 但在 eval_predictions.json 里的问题
        # 这些问题第一次就答错了，MPT = 0
        for idx, info in initial_results.items():
            if str(idx) not in mpt_data:
                # print(idx, end=" ")
                # 第一次就答错，MPT = 0
                subcategory = info["subcategory"]
                if subcategory:
                    stats[subcategory]["total_time"] += 0
                    stats[subcategory]["count"] += 1

    # 计算平均 MPT
    results = {}
    
    # 小类 MPT
    for subcategory in stats:
        total_time = stats[subcategory]["total_time"]
        count = stats[subcategory]["count"]
        avg_mpt = (total_time / count) if count > 0 else 0.0
        results[subcategory] = {
            "mpt": avg_mpt,
            "mpt(/min)": avg_mpt/60,
            "total_time": total_time,
            "count": count
        }
    
    # 大类 MPT
    for category, subcategories in CATEGORY_MAPPING.items():
        total_time = 0.0
        total_count = 0
        
        for subcat in subcategories:
            if subcat in stats:
                total_time += stats[subcat]["total_time"]
                total_count += stats[subcat]["count"]
        
        avg_mpt = (total_time / total_count) if total_count > 0 else 0.0
        results[category] = {
            "mpt": avg_mpt,
            "mpt(/min)": avg_mpt/60,
            "total_time": total_time,
            "count": total_count
        }
    
    # Overall MPT
    overall_time = sum(s["total_time"] for s in stats.values())
    overall_count = sum(s["count"] for s in stats.values())
    overall_mpt = (overall_time / overall_count) if overall_count > 0 else 0.0
    
    results["Overall"] = {
        "mpt": overall_mpt,
        "mpt(/min)": overall_mpt/60,
        "total_time": overall_time,
        "count": overall_count
    }

    # ========== 保存 MPT 结果为 JSON ==========
    mpt_json_path = os.path.join(BASE_DIR, f"mpt_{model_name}.json")
    with open(mpt_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f" MPT 结果已保存: {mpt_json_path}")
    
    return results




def main():
    
    print("=" * 80)
    print("开始计算 RTA 和 MPT 指标")
    print("=" * 80)
    print(f"\n基础目录: {BASE_DIR}\n")
    print(f"QA 目录: {QA_DIR}\n")
    
    all_rta_results = {}
    all_mpt_results = {}
    
    # 遍历每个模型
    for model, video_timestamps in MODEL_TIMESTAMPS.items():
        print(f"处理模型: {model}")
        print("-" * 80)
        
        # 收集该模型所有视频的 JSON 文件
        json_files = []
        for vid, timestamp in video_timestamps.items():
            # 根据是否指定时间戳来构建路径
            if timestamp:
                # 使用指定的时间戳
                file_path = os.path.join(BASE_DIR, model, vid, timestamp, "eval_predictions.json")
                if os.path.exists(file_path):
                    json_files.append(file_path)
                    print(f"找到 {file_path}")
                
        if not json_files:
            print(f"模型 {model} 没有找到任何结果文件，跳过")
            continue
        
        print(f"找到 {len(json_files)} 个视频的结果文件")
        
        # 计算 RTA 指标
        print(f"  【计算 RTA 指标】")
        rta_results = calculate_rta_for_model(model, json_files)
        all_rta_results[model] = rta_results
        
        overall_rta = rta_results.get("Overall", {}).get("accuracy", 0)
        overall_total = rta_results.get("Overall", {}).get("total", 0)
        overall_correct = rta_results.get("Overall", {}).get("correct", 0)
        print(f" RTA Overall: {overall_rta:.2f}% ({overall_correct}/{overall_total})\n")
        
        # 计算 MPT 指标
        print(f"  【计算 MPT 指标】")
        mpt_results = calculate_mpt_for_model(model, video_timestamps)
        all_mpt_results[model] = mpt_results
        
        overall_mpt = mpt_results.get("Overall", {}).get("mpt", 0)
        overall_count = mpt_results.get("Overall", {}).get("count", 0)
        print(f" MPT Overall: {overall_mpt:.2f}s (样本数: {overall_count})")
    
    if not all_rta_results:
        print("\n没有任何模型的结果，程序退出")
        return
    


if __name__ == "__main__":
    main()




