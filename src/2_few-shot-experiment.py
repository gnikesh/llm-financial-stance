"""
Few-Shot Stance Prediction Experiment
======================================

A single script that combines all few-shot experiment variants into one.
Use command-line arguments to control the experiment settings.

Usage Examples:
    # Basic run: no transcripts, most similar examples, sequential
    python few-shot-experiment.py --num_workers 1

    # Run with summarized transcripts and Chain-of-Thought prompts
    python few-shot-experiment.py --model gemma3 --transcripts Summarized --use_cot

    # Run with reasoning in few-shot examples, random sampling, parallel
    python few-shot-experiment.py --include_reasoning --few_shot_sampling random

    # Custom k values and targets
    python few-shot-experiment.py --model llama3-sdsc --k_values 1 5 --targets debt eps --num_runs 2
"""

import argparse
import os
import re
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import util
import openai

from embedding_client import EmbeddingClient


# ============================== ARGUMENT PARSING ==============================

def parse_arguments():
    """Parse command-line arguments for the experiment."""

    parser = argparse.ArgumentParser(
        description="Run few-shot stance prediction experiments with various LLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python few-shot-experiment.py
  python few-shot-experiment.py --model gemma3 --transcripts Summarized --use_cot
  python few-shot-experiment.py --include_reasoning
  python few-shot-experiment.py --model llama3-sdsc --k_values 1 5 --num_workers 8
        """,
    )

    # ----- Main Experiment Settings -----
    main_group = parser.add_argument_group("Main Settings")

    main_group.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini-2025-04-14",
        help=(
            "LLM model name for predictions. "
            "Examples: llama3-sdsc, gemma3, gpt-4.1-mini-2025-04-14, "
            "mistral-small:24b, phi4-reasoning:14b, qwen3:32b. "
            "Default: gpt-4.1-mini-2025-04-14."
        ),
    )

    main_group.add_argument(
        "--client-api-key", "--api-key", dest="client_api_key", default=None,
        help="API key for the LLM client. Default: use OPENAI_API_KEY.",
    )

    main_group.add_argument(
        "--client-base-url", "--base-url", dest="client_base_url", default=None,
        help="OpenAI-compatible API base URL. Default: standard OpenAI endpoint.",
    )

    main_group.add_argument(
        "--extract-model", default=None,
        help="Model used to extract the final stance through the same client. "
             "Default: use --model.",
    )

    main_group.add_argument(
        "--transcripts",
        type=str,
        default="No",
        choices=["No", "Full", "Summarized"],
        help=(
            "Whether to include transcript/SEC filing context in the prompt. "
            "No = no context, Full = full text, Summarized = summarized text. "
            "Default: No"
        ),
    )

    main_group.add_argument(
        "--use_cot",
        action="store_true",
        help="Enable Chain-of-Thought (CoT) reasoning prompts. Default: off",
    )

    main_group.add_argument(
        "--few_shot_sampling",
        type=str,
        default="most_similar",
        choices=["random", "most_similar"],
        help=(
            "How to select few-shot examples. "
            "'random' picks randomly, 'most_similar' picks by text similarity. "
            "Default: most_similar"
        ),
    )

    main_group.add_argument(
        "--include_reasoning",
        action="store_true",
        help=(
            "Include LLM reasoning alongside each few-shot example. "
            "When enabled, examples show the sentence + reasoning from the "
            "reasoning column. Default: off"
        ),
    )

    # ----- Experiment Parameters -----
    exp_group = parser.add_argument_group("Experiment Parameters")

    exp_group.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Number of experiment runs to average over. Default: 3",
    )

    exp_group.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=[0, 1, 5, 10],
        help=(
            "Number of few-shot examples per class (space-separated list). "
            "Default: 0 1 5 10"
        ),
    )

    exp_group.add_argument(
        "--targets",
        nargs="+",
        type=str,
        default=["debt", "eps", "sales"],
        help="Target variables to predict (space-separated). Default: debt eps sales",
    )

    exp_group.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers. Use 1 for sequential execution. Default: 4",
    )

    # ----- Path Settings -----
    path_group = parser.add_argument_group("Path Settings")

    path_group.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help=(
            "Directory to save results. If not provided, a directory name is "
            "automatically generated based on experiment settings."
        ),
    )

    path_group.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data', 'train-test-split'),
        help="Base directory containing the datasets (with ECT/SEC subdirectories containing train.csv/test.csv).",
    )

    path_group.add_argument(
        "--prompt_dir",
        type=str,
        default=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data', 'prompts'),
        help="Directory containing prompt template files.",
    )

    path_group.add_argument(
        "--context_data_dir",
        type=str,
        default=os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'data'),
        help="Base directory containing transcript/SEC context files (default: data/).",
    )

    # ----- Advanced Settings -----
    adv_group = parser.add_argument_group("Advanced Settings (usually no need to change)")

    adv_group.add_argument(
        "--embedding_backend",
        type=str,
        default="nautilus",
        choices=["local", "nautilus"],
        help="Embedding backend: 'nautilus' (uses NAUTILUS_API_KEY) or 'local' (localhost server). Default: nautilus",
    )

    adv_group.add_argument(
        "--embedding_model",
        type=str,
        default="qwen3-embedding",
        help="Embedding model name (only used with --embedding_backend nautilus). Default: qwen3-embedding",
    )

    adv_group.add_argument(
        "--embedding_url",
        type=str,
        default="http://localhost:8000",
        help="URL of the local embedding server (only used with --embedding_backend local). Default: http://localhost:8000",
    )

    adv_group.add_argument(
        "--nautilus_url",
        type=str,
        default="https://ellm.nrp-nautilus.io/v1",
        help="Nautilus API URL used by the embedding backend.",
    )

    adv_group.add_argument(
        "--stance_column",
        type=str,
        default="LLM_Stance_1",
        help="Column name for ground truth stance labels in the CSV. Default: LLM_Stance_1",
    )

    adv_group.add_argument(
        "--reasoning_column",
        type=str,
        default="LLM_Reason_1",
        help=(
            "Column name for LLM reasoning (used with --include_reasoning). "
            "Default: LLM_Reason_1"
        ),
    )

    adv_group.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt (useful for bash scripts).",
    )

    return parser.parse_args()


# ============================== GLOBAL STATE ==============================
# These variables are set once in main() and then used by all functions.
# In parallel mode, child processes inherit these via fork().

ARGS = None       # Parsed command-line arguments
SAVE_ROOT = None  # Directory where results are saved

# Mapping from short data type codes to full directory names
DATA_TYPES = {"ECT": "Earnings-Call-Transcript", "SEC": "SEC-DATA"}

# Maximum context length for vLLM-served models
MAX_CONTEXT_LENGTH = 8192
MAX_COMPLETION_TOKENS = 1024
MAX_PROMPT_TOKENS = MAX_CONTEXT_LENGTH - MAX_COMPLETION_TOKENS
# Approximate characters per token (conservative estimate)
CHARS_PER_TOKEN = 4

# The LLM client and embedding client are initialized once in main().
client = None
client_embed = None


def setup_clients(args):
    """Initialize the configured LLM client and the embedding client."""
    global client, client_embed

    client_options = {}
    if args.client_api_key is not None:
        client_options["api_key"] = args.client_api_key
    if args.client_base_url is not None:
        client_options["base_url"] = args.client_base_url

    # Without overrides, OpenAI uses its standard endpoint and OPENAI_API_KEY.
    client = openai.OpenAI(**client_options)

    if args.embedding_backend == "nautilus":
        client_embed = EmbeddingClient(
            backend="nautilus",
            model=args.embedding_model,
            nautilus_url=args.nautilus_url,
        )
    else:
        client_embed = EmbeddingClient(url=args.embedding_url)


def build_save_directory(args):
    """
    Automatically build a results directory name from the experiment settings.

    For example:
        few-shot-chain-of-thought-no-transcripts-most-similar-examples
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    base = os.path.join(project_root, 'experiments', 'results', 'few-shot')

    name_parts = ["few-shot"]

    if args.use_cot:
        name_parts.append("chain-of-thought")

    if args.transcripts == "No":
        name_parts.append("no-transcripts")
    elif args.transcripts == "Full":
        name_parts.append("with-transcripts")
    elif args.transcripts == "Summarized":
        name_parts.append("with-transcripts-summarized")

    if args.few_shot_sampling == "most_similar":
        name_parts.append("most-similar-examples")
    else:
        name_parts.append("random-examples")

    if args.include_reasoning:
        name_parts.append("with-reasoning")

    return os.path.join(base, "-".join(name_parts))


def print_config():
    """Print the experiment configuration so the user can verify settings."""
    print("=" * 60)
    print("  Few-Shot Stance Prediction Experiment")
    print("=" * 60)
    print(f"  Model:              {ARGS.model}")
    print(f"  Client URL:         {ARGS.client_base_url or 'OpenAI default'}")
    print(f"  Extract model:      {ARGS.extract_model or 'same as model'}")
    print(f"  Transcripts:        {ARGS.transcripts}")
    print(f"  Chain-of-Thought:   {'Yes' if ARGS.use_cot else 'No'}")
    print(f"  Sampling:           {ARGS.few_shot_sampling}")
    print(f"  Include Reasoning:  {'Yes' if ARGS.include_reasoning else 'No'}")
    print(f"  K values:           {ARGS.k_values}")
    print(f"  Targets:            {ARGS.targets}")
    print(f"  Num runs:           {ARGS.num_runs}")
    print(f"  Num workers:        {ARGS.num_workers}")
    print(f"  Stance column:      {ARGS.stance_column}")
    print(f"  Save directory:     {SAVE_ROOT}")
    print(f"  Data directory:     {ARGS.data_dir}")
    print("=" * 60)


# ============================== UTILITY FUNCTIONS ==============================


def estimate_tokens(text):
    """Estimate the number of tokens in a text string."""
    return len(text) // CHARS_PER_TOKEN


def truncate_to_fit(context_text, max_tokens):
    """Truncate context_text so it fits within max_tokens (estimated)."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(context_text) <= max_chars:
        return context_text
    truncated = context_text[:max_chars]
    print(f"Warning: Truncated context from ~{estimate_tokens(context_text)} to ~{max_tokens} tokens to fit context window.")
    return truncated


def get_transcript_context(data_type, company, quarter, year, filename):
    """
    Load transcript or SEC filing context to include in the prompt.

    Returns an empty string if transcripts are disabled (--transcripts No).
    Otherwise, reads the full or summarized context file from disk.
    """
    # If transcripts are disabled, return nothing
    if ARGS.transcripts == "No":
        return ""

    # ---------- Build the file path and context prefix ----------
    if data_type == "ECT":
        if ARGS.transcripts == "Full":
            path = (
                f"{ARGS.context_data_dir}/Earnings-Call-Transcript/call_transcripts/"
                f"{quarter}-{year}-{company}-Transcript.txt"
            )
            context_prefix = (
                "The entire earnings call transcript of the company is given below as the context. "
                "Please carefully analyze the context before making any decision. Based on the context provided below and the information from the text, "
                "classify the outlook of text for the given target. When providing the reason, please identify the specific section of the context that was helpful. "
                "Furthermore, state if the context is either helpful/necessary or not necessary for making the classification in your response.\n\n"
            )

        elif ARGS.transcripts == "Summarized":
            path = (
                f"{ARGS.context_data_dir}/Earnings-Call-Transcript/"
                f"call_transcripts-summarized-by-ChatGPT-o3/"
                f"{quarter}-{year}-{company}-Transcript.txt"
            )
            context_prefix = (
                "The summary of earnings call transcript of the company is given below as the context. "
                "Please carefully analyze the context before making any decision. Based on the context provided below and the information from the text, "
                "classify the outlook of text for the given target. "
                "Furthermore, state if the context is either helpful/necessary or not necessary for making the classification in your response.\n\n"
            )

    elif data_type == "SEC":
        if ARGS.transcripts == "Full":
            path = f"{ARGS.context_data_dir}/SEC-DATA/section-7-manually-extracted/{filename}"
            context_prefix = (
                "The Section 7 (Management's Discussion and Analysis of Financial Condition and Results of Operations.) section of 10-K report of the company is given below as the context. "
                "Please carefully analyze the context before making any decision. Based on the context provided below and the information from the text, "
                "classify the outlook of text for the given target. When providing the reason, please identify the specific section of the context that was helpful. "
                "Furthermore, state if the context is either helpful/necessary or not necessary for making the classification in your response.\n\n"
            )

        elif ARGS.transcripts == "Summarized":
            path = (
                f"{ARGS.context_data_dir}/SEC-DATA/"
                f"section-7-manually-extracted-summarized-by-ChatGPT-o3/{filename}"
            )
            context_prefix = (
                "The summary of Section 7 (Management's Discussion and Analysis of Financial Condition and Results of Operations.) section of 10-K report of the company is given below as the context. "
                "Please carefully analyze the context before making any decision. Based on the context provided below and the information from the text, "
                "classify the outlook of text for the given target. "
                "Furthermore, state if the context is either helpful/necessary or not necessary for making the classification in your response.\n\n"
            )

    else:
        raise ValueError(f"Unknown data type: {data_type}")

    # ---------- Read and return the context ----------
    with open(path, "r", errors="ignore") as f:
        context = f.read()

    return context_prefix + context


def get_prompt(data_type, text, target, quarter, year, company, filename, few_shot_examples):
    """
    Build the full prompt by combining a template, optional context, and examples.

    Uses Chain-of-Thought templates when --use_cot is enabled;
    otherwise uses the standard Few-shot templates.
    """
    # Select the right prompt template file
    if ARGS.use_cot:
        prompt_paths = {
            "ECT": os.path.join(ARGS.prompt_dir, "Chain-of-Thought_base_prompt_ECT.txt"),
            "SEC": os.path.join(ARGS.prompt_dir, "Chain-of-Thought_base_prompt_SEC.txt"),
        }
    else:
        prompt_paths = {
            "ECT": os.path.join(ARGS.prompt_dir, "Few-shot_ECT_base_prompt.txt"),
            "SEC": os.path.join(ARGS.prompt_dir, "Few-shot_SEC_base_prompt.txt"),
        }

    prompt_path = prompt_paths[data_type]

    # Read the prompt template
    with open(prompt_path, "r") as f:
        prompt_template = f.read()

    # Get transcript/SEC context (empty string if --transcripts No)
    context = get_transcript_context(data_type, company, quarter, year, filename)

    # Estimate tokens used by everything except context + examples,
    # then truncate context (and examples if needed) to fit within MAX_PROMPT_TOKENS.
    prompt_without_variable = prompt_template.format(
        target=target, text=text, transcript_context="", few_shot_examples="",
    )
    tokens_without_variable = estimate_tokens(prompt_without_variable)
    remaining_tokens = MAX_PROMPT_TOKENS - tokens_without_variable

    # Budget: give few-shot examples their space first, then context gets the rest
    examples_tokens = estimate_tokens(few_shot_examples)
    if examples_tokens > remaining_tokens:
        few_shot_examples = truncate_to_fit(few_shot_examples, remaining_tokens)
        examples_tokens = remaining_tokens
    context_budget = remaining_tokens - examples_tokens
    if context_budget > 0:
        context = truncate_to_fit(context, context_budget)
    else:
        context = ""

    # Fill in the template placeholders
    return prompt_template.format(
        target=target,
        text=text,
        transcript_context=context,
        few_shot_examples=few_shot_examples,
    )


def extract_final_stance(text, model_name, client):
    """
    Extract the final stance (Positive/Negative/Neutral) from an LLM's output.

    Uses the same model and client that generated the response to avoid
    overloading a single shared model when running experiments in parallel.
    Retries up to 10 times if the result is not a valid stance.
    """
    start_time = time.time()

    message = [
        {
            "role": "user",
            "content": (
                "Extract the final stance from the text. Do not try to guess the stance. "
                "The text might mention multiple stances, only extract the final, concluding "
                "stance mentioned in the text. Reply only Positive, Negative, or Neutral.\n"
                f'Here is the text:\n"{text}" '
            ),
        }
    ]

    content = ""
    for _ in range(10):
        # Keep retrying the API call until it succeeds
        while True:
            try:
                out = client.chat.completions.create(
                    model=model_name,
                    messages=message,
                )
                break
            except Exception as e:
                print(f"Error extracting stance: {e}. Retrying in 60s...")
                time.sleep(60)
                continue

        # Some OpenAI-compatible APIs may return a raw string.
        if isinstance(out, str):
            content = out.strip().lower().replace(".", "")
        else:
            content = out.choices[0].message.content.strip().lower().replace(".", "")

        # Check if we got a valid stance
        if content in ["positive", "negative", "neutral"]:
            print(f"Time taken to extract Stance (Seconds): {time.time() - start_time:.2f}")
            return content.capitalize()

    # After 10 attempts, return whatever we got
    time.sleep(1)
    print(f"Time taken to extract Stance (Seconds): {time.time() - start_time:.2f}")
    return content


def read_random_few_shot_examples(text, filepath, k, stance_column):
    """
    Randomly select k few-shot examples per stance class from the training data.

    When --include_reasoning is enabled, each example includes the LLM's reasoning
    and examples are interleaved across classes.
    When disabled, examples are grouped by class.

    Args:
        text:           Input text (unused for random sampling, kept for consistent API)
        filepath:       Path to the training CSV file
        k:              Number of examples to pick per class
        stance_column:  Column name for stance labels
    """
    if k == 0:
        return ""

    df = pd.read_csv(filepath)
    df.dropna(subset=[stance_column], inplace=True)
    text_col = "text" if "text" in df.columns else "Instances"

    if ARGS.include_reasoning:
        # ---- With reasoning: interleaved format ----
        # Collect examples along with their reasoning
        examples = {cat: [] for cat in ["Positive", "Negative", "Neutral"]}
        for _, row in df.iterrows():
            stance = row[stance_column]
            if stance in examples:
                examples[stance].append([row[text_col], row[ARGS.reasoning_column]])

        # Randomly pick k examples per class
        category_examples = {}
        for cat in ["Positive", "Negative", "Neutral"]:
            exs = examples[cat]
            if exs:
                num_to_select = min(k, len(exs))
                category_examples[cat] = random.sample(exs, num_to_select)
            else:
                category_examples[cat] = []

        # Interleave examples: one from each class, then repeat
        output = []
        max_examples = max(
            len(category_examples[cat]) for cat in ["Positive", "Negative", "Neutral"]
        )
        example_id = 0
        for example_idx in range(max_examples):
            for cat in ["Positive", "Negative", "Neutral"]:
                if example_idx < len(category_examples[cat]):
                    example_data = category_examples[cat][example_idx]
                    output.append(f"### Example {example_id + 1} - {cat}\n")
                    output.append(f"Sentence: {example_data[0]}\n")
                    output.append(f"Target: {cat}\n\n")
                    output.append(f"{example_data[1]}\n")
                    output.append("---\n")
                    example_id += 1

    else:
        # ---- Without reasoning: grouped by class ----
        examples = {"Positive": [], "Negative": [], "Neutral": []}
        for _, row in df.iterrows():
            stance = row[stance_column]
            if stance in examples:
                examples[stance].append(row[text_col])

        output = []
        for cat in ["Positive", "Negative", "Neutral"]:
            output.append(f"{cat}:\n{'-' * 20}\n")
            selected = examples[cat][:k]
            for idx, ex in enumerate(selected, 1):
                output.append(f"{idx}. {ex}\n")
            output.append("\n")

    return "Below are few examples:\n\n" + "".join(output)


def read_closest_few_shot_examples(text, filepath, k, stance_column):
    """
    Get k most similar few-shot examples per stance class using embeddings.

    When --include_reasoning is enabled, each example includes the LLM's reasoning
    and examples are interleaved across classes.
    When disabled, examples are grouped by class.

    Args:
        text:           Input text to find similar examples for
        filepath:       Path to the training CSV file
        k:              Number of examples to pick per class
        stance_column:  Column name for stance labels
    """
    if k == 0:
        return ""

    df = pd.read_csv(filepath)
    text_col = "text" if "text" in df.columns else "Instances"

    if ARGS.include_reasoning:
        # ---- With reasoning: interleaved format ----
        # Collect examples with their reasoning text
        examples = {cat: [] for cat in ["Positive", "Negative", "Neutral"]}
        for _, row in df.iterrows():
            stance = row[stance_column]
            if stance in examples:
                reason = row[ARGS.reasoning_column]
                sentence = row[text_col]
                # When no transcripts, clean out any "Context..." trailing text
                if ARGS.transcripts == "No":
                    cleaned_reason = re.sub(
                        r"\n?Context.*$", "", reason, flags=re.DOTALL
                    )
                else:
                    cleaned_reason = reason
                examples[stance].append([sentence, cleaned_reason])

        # Get embedding for the input text
        while True:
            try:
                text_emb = client_embed.get_embeddings([text])
                break
            except Exception as e:
                print(f"Error getting embeddings for input text: {e}. Retrying...")
                time.sleep(30)
                continue

        # Find the k most similar examples per class
        category_examples = {}
        for cat in ["Positive", "Negative", "Neutral"]:
            exs = examples[cat]
            if exs:
                example_texts = [ex[0] for ex in exs]
                while True:
                    try:
                        example_embs = client_embed.get_embeddings(example_texts)
                        break
                    except Exception as e:
                        print(f"Error getting embeddings for {cat} examples: {e}. Retrying...")
                        time.sleep(30)
                        continue

                sims = util.cos_sim(text_emb, example_embs)[0].cpu().numpy()
                topk_idx = np.argsort(-sims)[: min(k, len(exs))]
                category_examples[cat] = [(exs[i], sims[i]) for i in topk_idx]
            else:
                category_examples[cat] = []

        # Interleave examples across classes
        output = []
        max_examples = max(
            len(category_examples[cat]) for cat in ["Positive", "Negative", "Neutral"]
        )
        example_id = 0
        for example_idx in range(max_examples):
            for cat in ["Positive", "Negative", "Neutral"]:
                if example_idx < len(category_examples[cat]):
                    example_data, similarity_score = category_examples[cat][example_idx]
                    output.append(f"### Example {example_id + 1}\n")
                    output.append(f"Sentence: {example_data[0]}\n")
                    output.append(f"Target: {cat}\n\n")
                    output.append(f"{example_data[1]}\n")
                    output.append("---\n")
                    example_id += 1

    else:
        # ---- Without reasoning: grouped by class ----
        examples = {cat: [] for cat in ["Positive", "Negative", "Neutral"]}
        for _, row in df.iterrows():
            stance = row[stance_column]
            if stance in examples:
                examples[stance].append(row[text_col])

        # Get embedding for the input text
        while True:
            try:
                text_emb = client_embed.get_embeddings(text)
                break
            except Exception as e:
                print(f"Error getting embeddings: {e}. Retrying...")
                time.sleep(30)
                continue

        output = []
        for cat in ["Positive", "Negative", "Neutral"]:
            output.append(f"{cat}:\n{'-' * 20}\n")
            exs = examples[cat]
            if exs:
                while True:
                    try:
                        embs = client_embed.get_embeddings(exs)
                        break
                    except Exception as e:
                        print(f"Error getting embeddings: {e}. Retrying...")
                        time.sleep(30)
                        continue

                sims = util.cos_sim(text_emb, embs)[0].cpu().numpy()
                topk_idx = np.argsort(-sims)[:k]
                for idx, i in enumerate(topk_idx, 1):
                    output.append(f"{idx}. {exs[i]}\n")
            else:
                output.append("No guidelines available.\n")
            output.append("\n")

    return "Below are few examples:\n\n" + "".join(output)


def predict_stance(row, model_name, data_type, target, few_shot_examples):
    """
    Predict the stance for a single test instance.

    Builds the full prompt, sends it to the appropriate LLM API,
    and extracts the stance label from the response.
    """
    # Build the full prompt
    prompt = get_prompt(
        data_type,
        row["Instances"],
        target,
        row["Quarter"],
        row["Year"],
        row["Company"],
        row.get("filename", "") if "filename" in row else "",
        few_shot_examples,
    )

    # Small delay to avoid rate limiting
    if "gpt" in model_name:
        time.sleep(1)
    else:
        time.sleep(random.randint(1, 8))

    # Use the same user-configured client for generation and extraction.
    sleep_time = 60
    while True:
        try:
            chat = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_COMPLETION_TOKENS,
            )
            break
        except Exception as e:
            print(f"Error with {model_name}: {e}. Retrying in {sleep_time}s...")
            time.sleep(sleep_time)
            sleep_time = min(sleep_time * 2, 600)  # Exponential backoff, max 10 min
            continue

    # Some OpenAI-compatible APIs may return a raw string.
    if isinstance(chat, str):
        content = chat.strip()
    else:
        content = chat.choices[0].message.content.strip()

    stance = extract_final_stance(content, ARGS.extract_model or model_name, client)
    return pd.Series([stance, content])


# ============================== EXPERIMENT RUNNER ==============================


def run_experiment(model_name, target, data_type, k, run_id):
    """
    Run a single experiment: predict stances for all test instances and save results.

    This is the core function used by both sequential and parallel modes.

    Returns a dict with the experiment result (model, target, data_type, k, run_id, accuracy).
    """
    # Create the output directory for this model + data type
    result_dir = os.path.join(SAVE_ROOT, model_name, data_type)
    os.makedirs(result_dir, exist_ok=True)

    # Build the train/test file paths
    train_file = os.path.join(ARGS.data_dir, DATA_TYPES[data_type], target, "train.csv")
    test_file = os.path.join(ARGS.data_dir, DATA_TYPES[data_type], target, "test.csv")

    stance_col = ARGS.stance_column

    print(f"\nModel: {model_name}, Data: {data_type}, Target: {target}, k={k}, Run={run_id}")

    # Load test data
    test_df = pd.read_csv(test_file)
    test_df[stance_col] = test_df[stance_col].str.capitalize()

    # Pick the few-shot example function based on the sampling method
    if ARGS.few_shot_sampling == "most_similar":
        few_shot_func = read_closest_few_shot_examples
    else:
        few_shot_func = read_random_few_shot_examples

    # Run predictions with a progress bar
    tqdm.pandas(desc=f"{model_name}-{data_type}-{target}-k{k}-run{run_id}")
    test_df[["Pred_Stance", "Pred_Reason"]] = test_df.progress_apply(
        lambda row: predict_stance(
            row,
            model_name,
            data_type,
            target,
            few_shot_func(
                row["Instances"], train_file, k=k, stance_column=stance_col
            ),
        ),
        axis=1,
    )

    # Calculate accuracy
    accuracy = (test_df[stance_col] == test_df["Pred_Stance"]).mean() * 100

    # Save predictions to CSV
    pred_path = os.path.join(result_dir, f"{target}_k{k}_run{run_id}_predictions.csv")
    test_df.to_csv(pred_path, index=False)

    result = {
        "Model": model_name,
        "Target": target,
        "Data": data_type,
        "k": k,
        "Accuracy": accuracy,
        "run_id": run_id,
    }

    dashed = "-" * 50
    print(
        f"{dashed}\n"
        f"Completed: {model_name}-{data_type}-{target}-k{k}-run{run_id}, "
        f"Accuracy={accuracy:.2f}%\n"
        f"{dashed}"
    )

    return result


# ============================== MULTIPROCESSING ==============================


def experiment_worker(args_tuple):
    """
    Wrapper for run_experiment that catches exceptions.
    Used by the parallel executor so one failure doesn't crash everything.
    """
    model_name, target, data_type, k, run_id = args_tuple
    try:
        return run_experiment(model_name, target, data_type, k, run_id)
    except Exception as e:
        print(
            f"FAILED: {model_name}-{data_type}-{target}-k{k}-run{run_id}: {e}"
        )
        return {
            "Model": model_name,
            "Target": target,
            "Data": data_type,
            "k": k,
            "Accuracy": None,
            "run_id": run_id,
            "status": "failed",
            "error": str(e),
        }


def get_experiments_to_run():
    """
    Build the list of experiment combinations, skipping any that already have
    saved prediction files (so you can resume interrupted runs).

    Returns a list of (model_name, target, data_type, k, run_id) tuples.
    """
    experiments = []

    for run_id in range(1, ARGS.num_runs + 1):
        for k in ARGS.k_values:
            for target in ARGS.targets:
                for data_type in DATA_TYPES:
                    # Check if this experiment was already completed
                    pred_path = os.path.join(
                        SAVE_ROOT,
                        ARGS.model,
                        data_type,
                        f"{target}_k{k}_run{run_id}_predictions.csv",
                    )
                    if os.path.exists(pred_path):
                        print(
                            f"Skipping (already completed): "
                            f"{ARGS.model}-{data_type}-{target}-k{k}-run{run_id}"
                        )
                        continue

                    experiments.append(
                        (ARGS.model, target, data_type, k, run_id)
                    )

    return experiments


def save_results(new_results, existing_records, summary_path):
    """Save all results (existing + new) to the summary CSV file."""
    all_records = list(existing_records) + list(new_results)
    if all_records:
        df = pd.DataFrame(all_records)
        df.to_csv(summary_path, index=False)


# ============================== SEQUENTIAL MODE ==============================


def run_sequential():
    """Run experiments one at a time in a simple loop."""
    summary_path = os.path.join(SAVE_ROOT, f"all_results_summary_{ARGS.model}.csv")

    # Load any existing results (to resume interrupted runs)
    summaries = []
    if os.path.exists(summary_path):
        existing_df = pd.read_csv(summary_path)
        summaries = existing_df.to_dict(orient="records")
        print(f"Loaded {len(summaries)} existing results")

    # Get the list of experiments still to run
    experiments = get_experiments_to_run()

    if not experiments:
        print("All experiments already completed!")
        return

    print(f"\nRunning {len(experiments)} experiments sequentially...\n")

    for model_name, target, data_type, k, run_id in experiments:
        result = run_experiment(model_name, target, data_type, k, run_id)
        summaries.append(result)

        # Save after each experiment (so progress is never lost)
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(summary_path, index=False)

    print("\nDone! All results saved to:", summary_path)


# ============================== PARALLEL MODE ==============================


def run_parallel():
    """Run experiments in parallel using multiple worker processes."""
    summary_path = os.path.join(SAVE_ROOT, f"all_results_summary_{ARGS.model}.csv")

    # Load any existing results
    existing_records = []
    if os.path.exists(summary_path):
        existing_df = pd.read_csv(summary_path)
        existing_records = existing_df.to_dict(orient="records")
        print(f"Loaded {len(existing_records)} existing results")

    # Get the list of experiments still to run
    experiments = get_experiments_to_run()

    if not experiments:
        print("All experiments already completed!")
        return

    # Limit workers to number of experiments (no point having idle workers)
    max_workers = min(ARGS.num_workers, len(experiments))
    print(f"\nRunning {len(experiments)} experiments with {max_workers} parallel workers...\n")

    results = []
    failed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments to the pool
        future_to_args = {
            executor.submit(experiment_worker, args): args
            for args in experiments
        }

        # Collect results as they complete
        with tqdm(total=len(experiments), desc="Overall Progress") as pbar:
            for future in as_completed(future_to_args):
                try:
                    result = future.result()
                    results.append(result)

                    if result.get("status") == "failed":
                        failed_count += 1

                    pbar.set_postfix(
                        {"Done": len(results), "Failed": failed_count}
                    )
                    pbar.update(1)

                    # Save intermediate results every 10 completions
                    if len(results) % 10 == 0:
                        save_results(results, existing_records, summary_path)

                except Exception as e:
                    print(f"Unexpected error in future: {e}")
                    pbar.update(1)

    # Save final results
    save_results(results, existing_records, summary_path)

    completed = len(results) - failed_count
    print(f"\nDone! Completed: {completed}, Failed: {failed_count}")
    print(f"Results saved to: {summary_path}")


# ============================== MAIN ==============================


def main():
    global ARGS, SAVE_ROOT

    # Step 1: Parse command-line arguments
    ARGS = parse_arguments()

    # Step 2: Set up the LLM and embedding clients
    setup_clients(ARGS)

    # Step 3: Set up the save directory
    if ARGS.save_dir:
        SAVE_ROOT = ARGS.save_dir
    else:
        SAVE_ROOT = build_save_directory(ARGS)
    os.makedirs(SAVE_ROOT, exist_ok=True)

    # Step 4: Print the configuration
    print_config()

    # Step 5: Run experiments (sequential or parallel)
    if ARGS.num_workers <= 1:
        run_sequential()
    else:
        run_parallel()


if __name__ == "__main__":
    main()
