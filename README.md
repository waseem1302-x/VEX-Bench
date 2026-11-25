# VEX-Bench: Benchmarking Verification Complexity of LLM-Generated Misinformation

<div align="center">
  <a href="https://arxiv.org/abs/2609.35028"><img src="https://img.shields.io/badge/arXiv-2609.35028-b31b1b.svg?logo=arxiv" alt="arXiv:2609.35028"></a>
  <a href="https://huggingface.co/datasets/hanxunh/VEX-Bench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-VEX--Bench-blue.svg" alt="Hugging Face dataset"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/LICENSE-MIT-green.svg" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/made_with-Python-blue.svg" alt="Made with Python">
</div>

VEX-Bench evaluates what LLMs produce after misinformation-elicitation prompts, not only whether they comply. It scores five dimensions of verification burden—checkability, harm significance, source credibility signals, imposter legitimacy, and verification cost—across 7 models, 7 methods, 2 tasks, and 60 topics (5,880 conditions).

[Paper](https://openreview.net/forum?id=xYPvwioYRg) · [Interactive results](https://hanxunh.github.io/VEX-Bench/) · [Dataset](https://huggingface.co/datasets/hanxunh/VEX-Bench)

## Run the code

From the repository root:

```bash
pip install -r requirements.txt

# Inspect prompts without an API key or model call.
python benchmark/main.py --method direct --model openai/gpt-5.4 \
    --task fabrication --domain health --dry-run

# Generation and judging require OPENROUTER_API_KEY in the environment or .env.
python benchmark/main.py --method direct --model openai/gpt-5.4 \
    --task fabrication --domain health
python -m benchmark.run_judges --results-dir results/benchmark \
    --recursive --judges all
```

Generation writes to `results/benchmark/<model_slug>/<method>/<task>/<domain>__<topic>.json` (`/` becomes `__` in the model ID). The CC-Factcheck workflow is in [`cc-factcheck-skill/`](cc-factcheck-skill/).

## Rewrite sources and offline tests

The shipped `benchmark/source_articles.json` contains URLs only. Check coverage, then fetch missing sources:

```bash
python -m benchmark.fetch_sources --check
python -m benchmark.fetch_sources --fetch
```

The fetcher respects robots.txt, waits at least one second between requests, and saves extracted text and statuses in the gitignored `benchmark/source_articles.local.json`. It exits nonzero if any sources remain missing. The coverage report names blocked and failed URLs; a blocked status can also mean robots.txt was unavailable. Runs can still proceed for the sources fetched. `--fetch --force` retries every URL. The built-in HTML extraction is simple, so inspect source quality before experiments.

A missing-source warning means at least one rewrite condition has no source text; its count and first five topics refer to the requested slice. Run `python -m benchmark.fetch_sources --fetch` to clear it when all missing sources can be fetched, then use `--check` to inspect any remaining blocked or failed URLs. Fabrication does not require sources. Supplying source text changes rewrite inputs, the saved original-text field, and the factual reference available to JailNewsBench, so comparisons with earlier empty-source runs are affected.

From the repository root, run the offline suite with:

```bash
python -m unittest discover -s tests -v
```

It checks validator rules, JSON extraction strategies, the method registry and dry-run rendering, source fetching, and the missing-source guard. The suite uses local fixtures and does not perform generation, judging, or fact-checking; those workflows need API keys. Optionally install the dev-only `pytest` package and run `python -m pytest tests -v`.

**What still needs the lab:** generation and judging need an OpenRouter API key, the published Hugging Face datasets are separate from this repository, and the fact-checking agent and its workflow require their own resources and API access.

## Content Warning

This paper contains synthetic examples of potentially harmful content, including misinformation and persuasive narratives, generated solely for AI Safety research purposes. All prompts and instructions are constructed in controlled settings and are de-identified, excluding references to specific persons, institutions, or countries. Model-generated responses may nevertheless include references to real or fictional entities. Such references are studied as part of the model outputs and, where relevant to our analysis, are fact-checked and annotated for veracity. These materials are included solely for scientific analysis and do not reflect the authors’ views.

## Citation

```bibtex
@inproceedings{huang2026vexbench,
  title={VEX-Bench: Benchmarking Verification Complexity of LLM-Generated Misinformation},
  author={Huang, Hanxun and Wu, Yutao and Wang, Qizhou and Montaña-Niño, Silvia and Li, Yige and Zheng, Xiang and Doyuran, Elif Buse and Matich, Phoebe and Liu, Xiao and Ma, Xingjun and Erfani, Sarah and Leckie, Christopher},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems Evaluations and Datasets Track},
  year={2026},
  url={https://openreview.net/forum?id=xYPvwioYRg}
}
```
