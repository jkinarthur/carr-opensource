param(
    [Parameter(Mandatory=$true)]
    [string]$Ip,

    [Parameter(Mandatory=$true)]
    [string]$KeyPath,

    [Parameter(Mandatory=$true)]
    [ValidateSet("status", "stop-trainer", "start-trainer", "stop-instance")]
    [string]$Action,

    [string]$InstanceId = "",
    [string]$AwsCliPath = "C:\\Program Files\\Amazon\\AWSCLIV2\\aws.exe",
    [string]$RemoteRepo = "/home/ubuntu/carr-opensource",
    [string]$TrainArgs = "--n_epochs 200 --n_users 100000 --n_items 20000 --batch_size 512 --num_workers 4",
    [string]$ScreenName = "carr"
)

$ErrorActionPreference = "Stop"

switch ($Action) {
    "status" {
        $cmd = "set -e; cd $RemoteRepo; screen -ls || true; ps -ef | grep 'examples/mini_trainer.py' | grep -v grep || true; if [ -f outputs/train.log ]; then tail -n 20 outputs/train.log; fi"
        ssh -i $KeyPath ubuntu@$Ip "$cmd"
    }

    "stop-trainer" {
        $cmd = "set -e; cd $RemoteRepo; screen -S $ScreenName -X quit || true; pkill -f 'examples/mini_trainer.py' || true; sleep 1; screen -ls || true; ps -ef | grep 'examples/mini_trainer.py' | grep -v grep || true"
        ssh -i $KeyPath ubuntu@$Ip "$cmd"
    }

    "start-trainer" {
        $cmdTemplate = 'set -e; cd {0}; source .venv/bin/activate; last_ckpt=$(ls -1 outputs/checkpoints/ckpt_epoch*.pt 2>/dev/null | sort | tail -n 1 || true); if [ -n "$last_ckpt" ]; then resume_arg="--resume $last_ckpt"; else resume_arg=""; fi; screen -dmS {2} bash -lc ''cd {0}; source .venv/bin/activate; python -u examples/mini_trainer.py {1} $resume_arg | tee outputs/train.log''; sleep 2; screen -ls; ps -ef | grep ''examples/mini_trainer.py'' | grep -v grep || true'
        $cmd = [string]::Format($cmdTemplate, $RemoteRepo, $TrainArgs, $ScreenName)
        ssh -i $KeyPath ubuntu@$Ip "$cmd"
    }

    "stop-instance" {
        if (-not $InstanceId) {
            throw "InstanceId is required when Action=stop-instance"
        }
        & $AwsCliPath ec2 stop-instances --instance-ids $InstanceId
    }
}
