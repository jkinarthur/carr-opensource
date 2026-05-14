#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/carr-opensource
pkill -f "run_llm_api_resume.sh" || true
pkill -f "llm_baseline_evaluation.py --out_dir outputs/real_datasets_results/llm_baselines_api_resume" || true
sleep 1
ts=$(date +%Y%m%d_%H%M%S)
log=/home/ubuntu/carr-opensource/outputs/llm_api_resume_ml1m_${ts}.log
cat > /tmp/run_llm_api_resume_ml1m.sh << 'SH'
#!/usr/bin/env bash
set -euo pipefail
export OPENAI_API_KEY="$(tr -d "\r\n" < /tmp/openai_api_key.txt)"
cd /home/ubuntu/carr-opensource
mkdir -p outputs/real_datasets_results/llm_baselines_api_resume/ML-1M
echo "[API-LLM] dataset=ML-1M"
.venv/bin/python examples/llm_baseline_evaluation.py --out_dir "outputs/real_datasets_results/llm_baselines_api_resume/ML-1M" --max_batches 20 --models gpt-4o gpt-4o-mini
echo "[API-LLM] done=ML-1M"
SH
chmod +x /tmp/run_llm_api_resume_ml1m.sh
nohup bash /tmp/run_llm_api_resume_ml1m.sh > "$log" 2>&1 < /dev/null &
echo LOG:$log
echo PID:$!
sleep 2
ps -eo pid,etime,pcpu,pmem,cmd | grep -E "run_llm_api_resume_ml1m.sh|llm_baseline_evaluation.py|run_real_eval_remote.sh" | grep -v grep || true
echo --- LOG HEAD ---
head -n 40 "$log" || true
