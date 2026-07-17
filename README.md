# LLM Financial Stance

Code and data for evaluating LLM-based stance detection on financial targets (`debt`, `eps`, and `sales`) from earnings-call transcripts and SEC filings. The repository supports zero-shot, few-shot, and chain-of-thought (CoT) experiments.

## Setup

From the repository root, create a Python environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set the API key for the model provider you plan to use. For example:

```bash
export OPENAI_API_KEY="your-key"
export NAUTILUS_API_KEY="your-key"  # only for Nautilus-hosted models/embeddings
```

The full experiment runner also references local OpenAI-compatible LLM and embedding endpoints. Review the configuration block near the top of `src/run_all_zero_few_shot_CoT_experiment.sh` and adjust its models, URLs, worker count, and embedding backend for your environment.

## Run the experiments

Run one experiment family:

```bash
bash src/run_all_zero_few_shot_CoT_experiment.sh zero-shot
bash src/run_all_zero_few_shot_CoT_experiment.sh few-shot
bash src/run_all_zero_few_shot_CoT_experiment.sh cot
bash src/run_all_zero_few_shot_CoT_experiment.sh few-shot-cot
```

Run the complete suite with:

```bash
bash src/run_all_zero_few_shot_CoT_experiment.sh all
```

To run a smaller custom experiment, invoke a Python entry point directly. This example performs one zero-shot run using the standard OpenAI client:

```bash
python src/1_zero-shot-experiment.py \
  --mode no-transcript \
  --model gpt-4.1-mini-2025-04-14 \
  --num-runs 1
```

## Citation

If you use this repository, please cite:

```bibtex
@article{gyawali2025evaluating,
  title={Evaluating Large Language Models for Stance Detection on Financial Targets from SEC Filing Reports and Earnings Call Transcripts},
  author={Gyawali, Nikesh and Caragea, Doina and Vasenkov, Alex and Caragea, Cornelia},
  journal={arXiv preprint arXiv:2510.23464},
  year={2025}
}
```
