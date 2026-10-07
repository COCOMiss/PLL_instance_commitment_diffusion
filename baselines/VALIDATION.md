# Validation record

These are executable checks, not benchmark results on Tokyo/NYC/Gowalla.

- Environment: CPU, Python 3.12, PyTorch 2.14.1+cpu.
- `python -m unittest baselines.test_baselines -v`: 5 tests passed.
- Full-class block partition and gradients matched an independently constructed dense full-class scorer.
- PRODEN and PiCO: CLI training for two epochs each in `full` and `sampled` modes on a synthetic CSV/Qwen fixture.
- PRODEN and PiCO: optimizer/confidence/model state restored from `last.pt`, then an additional epoch executed.
- PiCO: standalone evaluation from `best.pt` executed; smoke-test provenance retained.
- Python compilation and `bash -n baselines/run_baselines.sh` passed.
- Multi-worker communication check is optional (`PLL_TEST_WORKERS=1`) and was not completed in this environment, whose local socket IPC is restricted. The CPU checks used `num_workers=0`.
- Full real-data training and CUDA execution were not run: the repository does not include the original Qwen `.pt` files.

No synthetic scores are included in the research result summary. Full/sampled experiments have separate output paths and are never aggregated together.
