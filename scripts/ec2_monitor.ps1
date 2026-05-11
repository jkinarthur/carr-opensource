param(
    [Parameter(Mandatory=$true)]
    [string]$Ip,

    [Parameter(Mandatory=$true)]
    [string]$KeyPath,

    [string]$RemoteRepo = "/home/ubuntu/carr-opensource"
)

$ErrorActionPreference = "Stop"

$remoteCmd = "set -e; cd $RemoteRepo; " +
"echo '=== screen sessions ==='; screen -ls || true; " +
"echo ''; echo '=== trainer processes ==='; ps -ef | grep 'examples/mini_trainer.py' | grep -v grep || true; " +
"echo ''; echo '=== latest train log ==='; if [ -f outputs/train.log ]; then tail -n 20 outputs/train.log; else echo 'outputs/train.log not found'; fi; " +
"echo ''; echo '=== latest checkpoints ==='; if [ -d outputs/checkpoints ]; then ls -lh outputs/checkpoints | tail -n 10; else echo 'outputs/checkpoints not found'; fi; " +
"echo ''; echo '=== gpu ==='; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader"

ssh -i $KeyPath ubuntu@$Ip "$remoteCmd"
