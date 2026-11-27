"""Small distributed bytewise regressions for BF16 and FP8 CPU checkpoint input."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('fp8', [False, True])
def test_cpu_staging_fetches_original_bytes(fp8):
    torch = pytest.importorskip('torch')
    pytest.importorskip('torch_npu')
    if not torch.npu.is_available() or torch.npu.device_count() < 2:
        pytest.skip('requires two available Ascend NPUs and a built DeepEP extension')
    script = Path(__file__).parent / 'engram/test_engram.py'
    command = [sys.executable, str(script), '--num-processes', '2',
               '--num-entries-per-layer', '256,257', '--hidden', '128',
               '--num-tokens', '8', '--num-entries-per-token', '4',
               '--cpu-staging', '--skip-performance']
    if fp8:
        command.append('--use-fp8')
    env = os.environ.copy()
    env['MASTER_PORT'] = env.get('MASTER_PORT', '29500')
    result = subprocess.run(command, cwd=script.parents[2], env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
