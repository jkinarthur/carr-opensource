# Step 2 + Step 3 Commands

Use these from the workspace root on Windows PowerShell.

## Step 2: One-command monitor

```powershell
./scripts/ec2_monitor.ps1 \
  -Ip 52.87.195.56 \
  -KeyPath C:\Users\jkina\.ssh\carr-v2-key-20260510-234402.pem
```

## Step 3: Safe control + resume

### Check status

```powershell
./scripts/ec2_run_control.ps1 \
  -Action status \
  -Ip 52.87.195.56 \
  -KeyPath C:\Users\jkina\.ssh\carr-v2-key-20260510-234402.pem
```

### Stop trainer cleanly (keeps instance running)

```powershell
./scripts/ec2_run_control.ps1 \
  -Action stop-trainer \
  -Ip 52.87.195.56 \
  -KeyPath C:\Users\jkina\.ssh\carr-v2-key-20260510-234402.pem
```

### Start/resume trainer from latest checkpoint

```powershell
./scripts/ec2_run_control.ps1 \
  -Action start-trainer \
  -Ip 52.87.195.56 \
  -KeyPath C:\Users\jkina\.ssh\carr-v2-key-20260510-234402.pem
```

### Stop the EC2 instance after stopping trainer

```powershell
./scripts/ec2_run_control.ps1 \
  -Action stop-instance \
  -InstanceId i-02b504ed0763f8396 \
  -Ip 52.87.195.56 \
  -KeyPath C:\Users\jkina\.ssh\carr-v2-key-20260510-234402.pem
```

## Recommended shutdown order

1. Run `status`.
2. Run `stop-trainer` and confirm no trainer process remains.
3. Wait for S3 checkpoint sync (if enabled).
4. Run `stop-instance`.
5. When you restart the instance, run `start-trainer`.
