#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/carr-opensource
if [ ! -s /tmp/openai_api_key.txt ]; then
  echo "ERROR: /tmp/openai_api_key.txt missing or empty"
  exit 1
fi
# Stop prior API-only resume runs to avoid duplicate API calls.
pkill -f "run_llm_api_resume.sh" || true
pkill -f "run_llm_api_resume_ml1m.sh" || true
pkill -f "llm_baseline_evaluation.py --out_dir outputs/real_datasets_results/llm_baselines_api_resume" || true
sleep 1

ts=$(date +%Y%m%d_%H%M%S)
log=/home/ubuntu/carr-opensource/outputs/llm_api_resume_all_${ts}.log
cat > /tmp/run_llm_api_resume.sh << 'SH'
#!/usr/bin/env bash
set -euo pipefail
export OPENAI_API_KEY="$(tr -d "\r\n" < /tmp/openai_api_key.txt)"
cd /home/ubuntu/carr-opensource
mkdir -p outputs/real_datasets_results/llm_baselines_api_resume
for d in ML-1M Beauty Toys Steam; do
  echo "[API-LLM] dataset=$d"
  .venv/bin/python examples/llm_baseline_evaluation.py --out_dir "outputs/real_datasets_results/llm_baselines_api_resume/$d" --max_batches 20 --models gpt-4o gpt-4o-mini
  echo "[API-LLM] done=$d"
done
echo "[API-LLM] all done"
SH
chmod +x /tmp/run_llm_api_resume.sh
nohup bash /tmp/run_llm_api_resume.sh > "$log" 2>&1 < /dev/null &
echo LOG:$log
echo PID:$!
sleep 2
ps -eo pid,etime,pcpu,pmem,cmd | grep -E "run_llm_api_resume.sh|llm_baseline_evaluation.py|run_real_eval_remote.sh" | grep -v grep || true
echo --- LOG HEAD ---
head -n 60 "$log" || true
