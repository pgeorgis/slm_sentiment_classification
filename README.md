## Introduction
This project contains an experiment using small language models (SLMs) to perform sentiment analysis on the IMDB movie review database. The goal is to automatically obtain binary classifications (positive or negative) for a given film review text.

Two SLMs are tested in this experiment:
- [Qwen2.5-0.5B](https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/blob/main/Qwen2.5-0.5B-Instruct-Q5_K_M.gguf) : a quantized version of Qwen2.5 with 500 million parameters
    
- [Qwen2.5-1.5B](https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/blob/main/Qwen2.5-1.5B-Instruct-Q5_K_M.gguf) : a quantized version of Qwen2.5 with 1.5 billion parameters

Four prompting techniques are tested:
- Zero-shot prompting
- Few-shot prompting
- Chain-of-thought prompting
- Prompt chaining with keyword-based sentiment analysis

Please see [analysis/study_summary.pdf](analysis/study_summary.pdf) for a full write-up on the experiments and detailed explanation about each of these prompt structures.


## Installation

Ensure that Python3 (my version is Python 3.12.0) is installed and run the following commands to create a virtual environment for the project and install dependencies into it:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Sentiment Analysis Experiment
To run the sentiment analysis experiment on a subset of the IMDB dataset, run the Python module [test_imdb_reviews.py](./test_imdb_reviews.py), optionally specifying a test size (number of movie reviews to select from test set partition) and test label.

Note that this test size N will select approximately N/2 reviews from each label (positive and negative), not N reviews from each label, i.e. with N = 500 the total sample size will be 500, not 1000.

The default test size is set to 500, which takes under 2 hours to test across all four prompt techniques on my local machine. All four prompts are tested using both the Qwen2.5-0.5B and Qwen2.5-1.5B small language models.

Example:

```
python3 test_imdb_reviews.py --test_size 500 --test_label replicate_results
```

Optionally the logging messages which would otherwise be printed to the console can be directed to a log file instead, e.g.:
```
mkdir -p logs

python3 test_imdb_reviews.py --test_size 500 --test_label replicate_results &> logs/2025-01-10_replicate-results.log &
```

## Results
An output directory will automatically be created under [results/](./results/) with the test date, optional test label, commit hash of the software version run, and an exact creation timestamp, e.g.:
[results/2025-01-09/official-test-n500/7373e83/2025-01-09_17-58-32](./results/2025-01-09/official-test-n500/7373e83/2025-01-09_17-58-32)

This output directory will contain the following TSV files:
- [results-summary.tsv](./results/2025-01-09/official-test-n500/7373e83/2025-01-09_17-58-32/results-summary.tsv)

    Summary of results aggregated by model and prompt technique.

- [imdb-sample-results.tsv](./results/2025-01-09/official-test-n500/7373e83/2025-01-09_17-58-32/imdb-sample-results.tsv)

    Detailed results of the tested sample of the IMDB dataset, including the full (preprocessed) review text of each test example, together with all true and predicted labels per model and prompt, as well as the extracted key words and phrases used for the prompt chaining method.

In addition, the output directory contains a [plots/](./results/2025-01-09/official-test-n500/7373e83/2025-01-09_17-58-32/plots/) subdirectory with several visualizations of results:
- [f1-comparison-by-model-and-prompt.png](./results/2025-01-09/official-test-n500/7373e83/2025-01-09_17-58-32/plots/f1-comparison-by-model-and-prompt.png)

    Bar graph plot displaying performance per model and prompt technique evaluated with F1 score.

- [f1-vs-latency-scatterplot.png](./results/2025-01-09/official-test-n500/7373e83/2025-01-09_17-58-32/plots/f1-vs-latency-scatterplot.png)

    Scatterplot with latency on the X-axis and F1 score on the Y-axis, illustrating the tradeoff between speed and accuracy per model and prompt technique.

Within the [plots/](./results/2025-01-09/official-test-n500/7373e83/2025-01-09_17-58-32/plots/) subdirectory are further subdirectories containing visualizations specific to each model-prompt pair, e.g. [Qwen0.5B with zero-shot prompting](./results/2025-01-09/official-test-n500/7373e83/2025-01-09_17-58-32/plots/Qwen-0.5B/zeroshot/):
- [confusion-matrix.png](./results/2025-01-09/official-test-n500/7373e83/2025-01-09_17-58-32/plots/Qwen-0.5B/zeroshot/confusion-matrix.png)

    Confusion matrix which displays numbers of true positives, true negatives, false positives, and false negatives per prompt and model.

- [wordcount-histogram.png](.results/2025-01-09/official-test-n500/7373e83/2025-01-09_17-58-32/plots/Qwen-0.5B/zeroshot/wordcount-histogram.png)

    Histogram showing the distribution of true positives, true negatives, false positives, and false negatives for the model-prompt pair in question across reviews of various lengths.
