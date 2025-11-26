# Lambda Labs Training Session

## Instance Details
- **Instance ID**: 7d5456a9732c4dcfa0422a1098e1f417
- **Instance Name**: ouroboros-training
- **IP Address**: 129.80.160.225
- **Region**: us-east-1 (Virginia, USA)
- **Instance Type**: gpu_1x_a100_sxm4
- **GPU**: NVIDIA A100-SXM4-40GB
- **Price**: $1.29/hr
- **Started**: 2025-11-26

## SSH Access
```bash
ssh lambda
```

SSH config (`~/.ssh/config`) optimized with:
- Connection multiplexing (first: ~600ms, subsequent: ~80ms)
- Auto-reconnect keepalive
- No host key warnings (Lambda IPs change frequently)

## Guest Agent
Installed Lambda Guest Agent for metrics visibility in Lambda Cloud console:
```bash
# Check status
ssh lambda "sudo systemctl status lambda-guest-agent"
```

## API Key
Stored in `.lambda_api_key` (gitignored)

## Training Location
```bash
cd ~/Fractal/research-log/phase7-ouroboros
```

## Check Training Status
```bash
# View training log
tail -f training.log

# Check GPU usage
nvidia-smi

# Check process
ps aux | grep train.py
```

## Training Config
- Model: 12 layers, 512 dim, 8 heads (~50M params)
- Batch size: 64
- Max iters: 5000
- Learning rate: 3e-4
- Data: 15,190 train samples, 2,696 val samples (balanced 50/50)

## Expected Duration
~2-3 hours for full training

## Download Results
```bash
scp lambda:~/Fractal/research-log/phase7-ouroboros/checkpoints/ckpt.pt ./checkpoints/
```

## Terminate Instance
```bash
python lambda_helper.py terminate
```

Or via API:
```bash
curl -X POST https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
  -H "Authorization: Bearer $(cat .lambda_api_key)" \
  -H "Content-Type: application/json" \
  -d '{"instance_ids": ["7d5456a9732c4dcfa0422a1098e1f417"]}'
```
