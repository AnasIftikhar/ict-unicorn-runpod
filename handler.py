"""
RunPod serverless handler — entry point for the pod.
Receives config JSON, runs optimization, returns results.
"""

import runpod
from optimize_core import run_optimization


def handler(job):
    try:
        config = job['input']
        return run_optimization(config)
    except Exception as e:
        return {"status": "error", "error": str(e)}


runpod.serverless.start({"handler": handler})
