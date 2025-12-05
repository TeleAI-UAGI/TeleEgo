
eval_videochat_online() {
  CUDA_VISIBLE_DEVICES=$1  python evaluate_videochat_online.py \
  --p_num "1" --qa_suffix "A" \
  --checkpoint "/gemini/space/model_zoo/VideoChat-Online" \
  --base_dir "./teleego_data" --seed 42 \
  --decision_window 5.0 --recall_delay 60.0 --max_recall_rounds 10 \
  --api_key "" --api_version "2025-01-01-preview" --end_point "" --engine "4o" \
  --max_new_tokens 32
  }


eval_qwen25_omni() {
  CUDA_VISIBLE_DEVICES=$1  python evaluate_qwen25_omni.py \
  --p_num "1" --qa_suffix "A" \
  --checkpoint "/gemini/space/model_zoo/Qwen2.5-Omni-7B" \
  --base_dir "./teleego_data" --seed 42 \
  --decision_window 5.0 --recall_delay 60.0 --max_recall_rounds 10 \
  --api_key "" --api_version "2025-01-01-preview" --end_point "" --engine "4o" \
  --max_new_tokens 32
  }


eval_qwen25_vl() {
  CUDA_VISIBLE_DEVICES=$1  python evaluate_qwen25_vl.py \
  --p_num "1" --qa_suffix "A" \
  --checkpoint "/gemini/space/model_zoo/Qwen2.5-VL-7B-Instruct" \
  --base_dir "./teleego_data" --seed 42 \
  --decision_window 5.0 --recall_delay 60.0 --max_recall_rounds 10 \
  --api_key "" --api_version "2025-01-01-preview" --end_point "" --engine "4o" \
  --max_new_tokens 32
  }


eval_minicpm_o() {
  CUDA_VISIBLE_DEVICES=$1  python evaluate_minicpm_o.py \
  --p_num "1" --qa_suffix "A" \
  --checkpoint "/gemini/space/model_zoo/MiniCPM-o-2_6" \
  --base_dir "./teleego_data" --seed 42 \
  --decision_window 5.0 --recall_delay 60.0 --max_recall_rounds 10 \
  --api_key "" --api_version "2025-01-01-preview" --end_point "" --engine "4o" \
  --max_new_tokens 32
  }



eval_gpt_4o() {
  CUDA_VISIBLE_DEVICES=$1  python evaluate_gpt_4o.py \
  --p_num "1" --qa_suffix "A" \
  --base_dir "./teleego_data" --seed 42 \
  --decision_window 5.0 --recall_delay 60.0 --max_recall_rounds 10 \
  --api_key "" \
  --api_version "2025-01-01-preview" --engine "gpt-4o" \
  --end_point "" \
  --max_new_tokens 32 \
  --max_retries 1 --retry_delay 10
  }


eval_gemini25_pro() {
  CUDA_VISIBLE_DEVICES=$1  python evaluate_gemini25_pro.py \
  --p_num "1" --qa_suffix "A" \
  --base_dir "./teleego_data" --seed 42 \
  --decision_window 5.0 --recall_delay 60.0 --max_recall_rounds 10 \
  --api_key "" \
  --api_version "2025-01-01-preview" --engine "gpt-4o" \
  --end_point "" \
  --max_new_tokens 32 \
  --max_retries 1 --retry_delay 10
  }


"$@"