"""
Zero-Shot Experiment Script
===========================
This script runs zero-shot stance classification experiments.

Modes (context):
  - no-transcript: No transcript/SEC context in the prompt
  - summarized:    Summarized transcript/SEC context
  - full:          Full transcript/SEC context

Optionally enable Chain-of-Thought (CoT) prompting with --use-cot.

Usage Examples:
--------------
# Run zero-shot with no context
python 1_zero-shot-experiment.py --mode no-transcript

# Run zero-shot with CoT prompting
python 1_zero-shot-experiment.py --mode no-transcript --use-cot

# Run with summarized context, 5 runs
python 1_zero-shot-experiment.py --mode summarized --num-runs 5

# Run with full context, specific models and targets
python 1_zero-shot-experiment.py --mode full --model llama3-sdsc --targets debt sales

# Run against an OpenAI-compatible custom endpoint
python 1_zero-shot-experiment.py --mode no-transcript --model llama3 \
    --client-base-url http://localhost:8005/v1 --client-api-key not-needed

# Specify custom output and data directories
python 1_zero-shot-experiment.py --mode summarized --results-dir /path/to/output --data-dir /path/to/data
"""

import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from sklearn.metrics import classification_report

# Enable progress bars for pandas operations
tqdm.pandas()

# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================
# Project root directory (parent of experiments/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Maximum context length for vLLM-served models
MAX_CONTEXT_LENGTH = 8192
MAX_COMPLETION_TOKENS = 1024
MAX_PROMPT_TOKENS = MAX_CONTEXT_LENGTH - MAX_COMPLETION_TOKENS
# Approximate characters per token (conservative estimate)
CHARS_PER_TOKEN = 4

# Available targets for stance classification
TARGETS = ['debt', 'eps', 'sales']

# Default data directory (new dataset)
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'train-test-split')

# Default results directory
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'results', 'zero-shot')

# Prompt directory
PROMPT_DIR = os.path.join(PROJECT_ROOT, 'data', 'prompts')

# Transcript directories (for full/summarized context modes)
ECT_TRANSCRIPTS_SUMMARIZED_DIR = os.path.join(PROJECT_ROOT, 'data', 'Earnings-Call-Transcript', 'call_transcripts-summarized-by-ChatGPT-o3')
ECT_TRANSCRIPTS_FULL_DIR = os.path.join(PROJECT_ROOT, 'data', 'Earnings-Call-Transcript', 'call_transcripts')
SEC_SECTION7_SUMMARIZED_DIR = os.path.join(PROJECT_ROOT, 'data', 'SEC-DATA', 'section-7-manually-extracted-summarized-by-ChatGPT-o3')
SEC_SECTION7_FULL_DIR = os.path.join(PROJECT_ROOT, 'data', 'SEC-DATA', 'section-7-manually-extracted')

# ============================================================================
# API CLIENT SETUP
# ============================================================================
def create_client(api_key=None, base_url=None):
    """Create the single OpenAI-compatible client used by the experiment."""
    client_options = {}
    if api_key is not None:
        client_options['api_key'] = api_key
    if base_url is not None:
        client_options['base_url'] = base_url

    # With no overrides, OpenAI() uses the standard OpenAI endpoint and the
    # OPENAI_API_KEY environment variable (the default ChatGPT configuration).
    return OpenAI(**client_options)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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


def get_prompt(text, target, data, mode, use_cot, quarter=None, year=None, company=None, filename=''):
    """
    Generate the prompt for stance classification based on the mode.

    Parameters:
    -----------
    text : str
        The text instance to classify
    target : str
        The target for classification (debt, eps, or sales)
    data : str
        Type of data ('ECT' for earnings call transcript or 'SEC' for SEC filing)
    mode : str
        Experiment mode ('no-transcript', 'summarized', or 'full')
    use_cot : bool
        Whether to use Chain-of-Thought prompt templates
    quarter : str, optional
        Quarter of the data (needed for 'summarized' and 'full' modes)
    year : str, optional
        Year of the data (needed for 'summarized' and 'full' modes)
    company : str, optional
        Company name (needed for 'summarized' and 'full' modes)
    filename : str, optional
        Filename for SEC data (needed for SEC data in 'summarized' and 'full' modes)

    Returns:
    --------
    str
        The complete prompt with or without transcript context
    """
    # Select prompt template based on data type and CoT setting
    if use_cot:
        prompt_paths = {
            'ECT': os.path.join(PROMPT_DIR, 'Chain-of-Thought_base_prompt_ECT.txt'),
            'SEC': os.path.join(PROMPT_DIR, 'Chain-of-Thought_base_prompt_SEC.txt'),
        }
    else:
        prompt_paths = {
            'ECT': os.path.join(PROMPT_DIR, 'ECT_base_prompt.txt'),
            'SEC': os.path.join(PROMPT_DIR, 'SEC_base_prompt.txt'),
        }

    prompt_path = prompt_paths[data]

    # Initialize transcript context as empty
    transcript_context = ""

    # If mode requires transcripts, load them
    if mode in ['summarized', 'full']:
        is_summarized = (mode == 'summarized')

        # Get transcript path based on data type and summarization preference
        if data == 'ECT':
            transcript_filename = f"{quarter}-{year}-{company}-Transcript.txt"
            if is_summarized:
                transcript_path = os.path.join(ECT_TRANSCRIPTS_SUMMARIZED_DIR, transcript_filename)
            else:
                transcript_path = os.path.join(ECT_TRANSCRIPTS_FULL_DIR, transcript_filename)

        elif data == 'SEC':
            if is_summarized:
                transcript_path = os.path.join(SEC_SECTION7_SUMMARIZED_DIR, filename)
            else:
                transcript_path = os.path.join(SEC_SECTION7_FULL_DIR, filename)

        # Read the transcript file
        try:
            with open(transcript_path, 'r', errors='ignore') as file:
                transcript_content = file.read()
        except FileNotFoundError:
            print(f"Warning: Transcript file not found: {transcript_path}")
            transcript_content = ""

        # Format the transcript context with appropriate description
        summary_text = " summarized " if is_summarized else " "

        if data == 'ECT':
            transcript_context = (
                f"The{summary_text}earnings call transcript of the company is given below as the context. "
                f"Please carefully analyze the context before making any decision. "
                f"Based on the context provided below and the information from the text, classify the outlook of text for the given target. "
                f"When providing the reason, please identify the specific section of the context that was helpful. "
                f"Furthermore, state if the context is either helpful/necessary or not necessary for making the classification in your response.\n\n"
                f"{transcript_content}"
            )
        elif data == 'SEC':
            transcript_context = (
                f"The{summary_text}Section 7 (Management's Discussion and Analysis of Financial Condition and Results of Operations.) "
                f"section of 10-K report of the company is given below as the context. "
                f"Please carefully analyze the context before making any decision. "
                f"Based on the context provided below and the information from the text, classify the outlook of text for the given target. "
                f"When providing the reason, please identify the specific section of the context that was helpful. "
                f"Furthermore, state if the context is either helpful/necessary or not necessary for making the classification in your response.\n\n"
                f"{transcript_content}"
            )

    # Read the base prompt template
    with open(prompt_path, 'r') as file:
        prompt = file.read()

    # Estimate tokens used by everything except transcript_context,
    # then truncate transcript_context to fit within MAX_PROMPT_TOKENS.
    if use_cot:
        prompt_without_context = prompt.format(target=target, text=text, transcript_context="", few_shot_examples="")
    else:
        prompt_without_context = prompt.format(target=target, text=text, transcript_context="")
    tokens_without_context = estimate_tokens(prompt_without_context)
    remaining_tokens = MAX_PROMPT_TOKENS - tokens_without_context
    if remaining_tokens > 0:
        transcript_context = truncate_to_fit(transcript_context, remaining_tokens)
    else:
        # No room for context at all — the template + text already fills the budget
        transcript_context = ""

    # Format the prompt with target, text, and transcript context
    # CoT templates also have {few_shot_examples} placeholder — pass empty for zero-shot
    if use_cot:
        prompt = prompt.format(target=target, text=text, transcript_context=transcript_context, few_shot_examples="")
    else:
        prompt = prompt.format(target=target, text=text, transcript_context=transcript_context)

    return prompt


def extract_stance(text, model_name, client):
    """
    Extract the final stance (Positive, Negative, or Neutral) from LLM response text.

    Uses the same model and client that generated the response to avoid
    overloading a single shared model when running experiments in parallel.

    Parameters:
    -----------
    text : str
        The response text from the LLM
    model_name : str
        Name of the model to use for extraction
    client : OpenAI
        API client to use for the extraction call

    Returns:
    --------
    str
        Extracted stance: 'Positive', 'Negative', or 'Neutral'
    """
    start_time = time.time()

    message = [
        {
            'role': 'user',
            'content': f"""Extract the final stance from the text. Do not try to guess the stance. The text might mention multiple stances, only extract the final, concluding stance mentioned in the text. Reply only Positive, Negative, or Neutral.
            Here is the text:
            "{text}" """
        }
    ]


    for attempt in range(10):
        while True:
            try:
                outputs = client.chat.completions.create(
                    model=model_name,
                    messages=message
                )
                break
            except Exception as e:
                print(f"Error in extracting stance: {e}. Retrying in 60 seconds...")
                time.sleep(60)
                continue

        # Some OpenAI-compatible APIs may return a raw string.
        if isinstance(outputs, str):
            content = outputs.lower().replace('.', '').strip()
        else:
            content = outputs.choices[0].message.content.lower().replace('.', '').strip()

        if content in ['positive', 'negative', 'neutral']:
            print(f"Time taken to extract Stance (Seconds): {time.time() - start_time:.2f}")
            return content.capitalize()

        print(f"Attempt {attempt}: Invalid stance '{content}', retrying...")

    print("Too many attempts to get a valid stance. Using same content")
    print(f"Time taken to extract Stance (Seconds): {time.time() - start_time:.2f}")
    return content.capitalize()


def get_stance(
    text, target, data, model_name, mode, use_cot, client, extract_model=None,
    quarter=None, year=None, company=None, filename=''
):
    """
    Get stance prediction from the LLM for a given text instance.

    Returns:
    --------
    tuple
        (stance, full_response) - The predicted stance and the full LLM response
    """
    prompt = get_prompt(text, target, data, mode, use_cot, quarter, year, company, filename)

    conversation_history = [
        {
            'role': 'user',
            'content': prompt
        }
    ]

    # The same user-configured client is used for every model.
    while True:
        try:
            outputs = client.chat.completions.create(
                model=model_name,
                messages=conversation_history,
                max_tokens=MAX_COMPLETION_TOKENS,
            )

            # Some OpenAI-compatible APIs may return a raw string.
            if isinstance(outputs, str):
                content = outputs
            else:
                content = outputs.choices[0].message.content

            # Detect HTML error pages from the gateway
            if not content or (isinstance(content, str) and content.strip().startswith('<!DOCTYPE')):
                raise ValueError("Model returned empty or HTML content, retrying...")

            break
        except Exception as e:
            print(f"Error in getting stance: {e}. Retrying in 60 seconds...")
            time.sleep(60)
            continue

    # The extraction model must be available through the same client.
    stance = extract_stance(content, extract_model or model_name, client)

    return stance, content


def find_agreement_percentage(predictions, labels):
    """
    Calculate the percentage of agreement between predictions and labels.
    """
    assert len(predictions) == len(labels), "Length of predictions and labels must be the same."
    agreement_count = sum(p == l for p, l in zip(predictions, labels))
    return round((agreement_count / len(predictions)) * 100, 2)


def print_classification_report(y_true, y_pred):
    """
    Generate a classification report as a DataFrame.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    return pd.DataFrame(report).T


# ============================================================================
# MAIN EXPERIMENT FUNCTION
# ============================================================================

def predict_dataframe(
    df, target, data_type, model_name, mode, use_cot, num_workers, label,
    client, extract_model=None
):
    """
    Run stance predictions on all rows in a DataFrame.

    When num_workers=1, processes rows sequentially (original behavior).
    When num_workers>1, processes rows in parallel using threads.

    Parameters:
    -----------
    df : pd.DataFrame
        Test data (must have 'Instances' and optionally 'Quarter', 'Year', 'Company', 'filename')
    target : str
        Target for classification (debt, eps, or sales)
    data_type : str
        'ECT' or 'SEC'
    model_name : str
        Name of the model to use
    mode : str
        Experiment mode ('no-transcript', 'summarized', or 'full')
    use_cot : bool
        Whether to use Chain-of-Thought prompting
    num_workers : int
        Number of parallel threads (1 = sequential)
    label : str
        Label for the progress bar
    client : OpenAI
        Client used for generation and stance extraction
    extract_model : str, optional
        Model used to extract the final stance (default: model_name)

    Returns:
    --------
    tuple
        (list of stances, list of reasons)
    """
    stances = [None] * len(df)
    reasons = [None] * len(df)

    def process_row(idx_row):
        idx, row = idx_row
        if mode == 'no-transcript':
            return idx, get_stance(
                row['Instances'], target, data_type, model_name, mode, use_cot,
                client, extract_model
            )
        else:
            return idx, get_stance(
                row['Instances'], target, data_type, model_name, mode, use_cot,
                client, extract_model, row['Quarter'], row['Year'], row['Company'],
                row.get('filename', '')
            )

    rows = list(df.iterrows())

    if num_workers <= 1:
        # Sequential mode: simple loop with progress bar
        for idx, row in tqdm(rows, desc=label):
            _, (stance, reason) = process_row((idx, row))
            stances[rows.index((idx, row))] = stance
            reasons[rows.index((idx, row))] = reason
    else:
        # Parallel mode: use threads (not processes) since work is I/O-bound (API calls)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_row, (idx, row)): i for i, (idx, row) in enumerate(rows)}
            with tqdm(total=len(rows), desc=label) as pbar:
                for future in as_completed(futures):
                    pos = futures[future]
                    _, (stance, reason) = future.result()
                    stances[pos] = stance
                    reasons[pos] = reason
                    pbar.update(1)

    return stances, reasons


def run_experiment(
    model_name, target, run_id, mode, use_cot, results_dir, data_dir,
    stance_column, num_workers, client, extract_model=None
):
    """
    Run a single experiment with specified parameters.

    Parameters:
    -----------
    model_name : str
        Name of the model to use
    target : str
        Target for classification (debt, eps, or sales)
    run_id : int
        Run identifier (for multiple runs)
    mode : str
        Experiment mode ('no-transcript', 'summarized', or 'full')
    use_cot : bool
        Whether to use Chain-of-Thought prompting
    results_dir : str
        Directory to save results
    data_dir : str
        Base directory for the dataset
    stance_column : str
        Column name for ground truth stance labels
    num_workers : int
        Number of parallel threads for API calls
    client : OpenAI
        Client used for generation and stance extraction
    extract_model : str, optional
        Model used to extract the final stance (default: model_name)

    Returns:
    --------
    list
        Summary of results for this run
    """
    cot_label = "CoT" if use_cot else "NoCoT"
    print(f"\n{'='*70}")
    print(f"Running: {model_name} | Target: {target} | Run: {run_id} | Mode: {mode} | {cot_label}")
    print(f"{'='*70}\n")

    run_summary = []

    # Create output directories upfront
    ect_output_dir = os.path.join(results_dir, model_name, 'ECT')
    sec_output_dir = os.path.join(results_dir, model_name, 'SEC')
    os.makedirs(ect_output_dir, exist_ok=True)
    os.makedirs(sec_output_dir, exist_ok=True)

    ect_output_file = os.path.join(ect_output_dir, f'test-ECT-{target}-{model_name}-run_{run_id}.csv')
    sec_output_file = os.path.join(sec_output_dir, f'test-SEC-{target}-{model_name}-run_{run_id}.csv')

    # --- ECT predictions ---
    if os.path.exists(ect_output_file):
        # ECT already done from a previous run — load results instead of re-running
        print(f"\nSkipping ECT (already completed): {ect_output_file}")
        ect_test_df = pd.read_csv(ect_output_file)
        ect_test_df[stance_column] = ect_test_df[stance_column].str.capitalize()
        ect_test_df['Pred_Stance'] = ect_test_df['Pred_Stance'].str.capitalize()
    else:
        ect_test_path = os.path.join(data_dir, 'Earnings-Call-Transcript', target, 'test.csv')
        print(f"\nLoading ECT data from: {ect_test_path}")
        ect_test_df = pd.read_csv(ect_test_path)
        ect_test_df[stance_column] = ect_test_df[stance_column].str.capitalize()

        print(f"Processing ECT data ({len(ect_test_df)} instances) with {num_workers} worker(s)...")
        ect_test_df['Pred_Stance'], ect_test_df['Pred_Reason'] = predict_dataframe(
            ect_test_df, target, 'ECT', model_name, mode, use_cot, num_workers,
            label=f"{model_name} ECT-{target}", client=client,
            extract_model=extract_model
        )
        # Save ECT immediately after completion
        ect_test_df.to_csv(ect_output_file, index=False)
        print(f"Saved ECT results to: {ect_output_file}")

    ect_agreement = find_agreement_percentage(ect_test_df['Pred_Stance'], ect_test_df[stance_column])
    run_summary.append({
        'Model': model_name, 'Target': target, 'Data': 'ECT',
        'Accuracy': ect_agreement, 'run_id': run_id, 'mode': mode, 'cot': use_cot
    })
    print(f"ECT Accuracy: {ect_agreement}%")

    # --- SEC predictions ---
    if os.path.exists(sec_output_file):
        # SEC already done from a previous run — load results instead of re-running
        print(f"\nSkipping SEC (already completed): {sec_output_file}")
        sec_test_df = pd.read_csv(sec_output_file)
        sec_test_df[stance_column] = sec_test_df[stance_column].str.capitalize()
        sec_test_df['Pred_Stance'] = sec_test_df['Pred_Stance'].str.capitalize()
    else:
        sec_test_path = os.path.join(data_dir, 'SEC-DATA', target, 'test.csv')
        print(f"\nLoading SEC data from: {sec_test_path}")
        sec_test_df = pd.read_csv(sec_test_path)
        sec_test_df[stance_column] = sec_test_df[stance_column].str.capitalize()

        print(f"Processing SEC data ({len(sec_test_df)} instances) with {num_workers} worker(s)...")
        sec_test_df['Pred_Stance'], sec_test_df['Pred_Reason'] = predict_dataframe(
            sec_test_df, target, 'SEC', model_name, mode, use_cot, num_workers,
            label=f"{model_name} SEC-{target}", client=client,
            extract_model=extract_model
        )
        # Save SEC immediately after completion
        sec_test_df.to_csv(sec_output_file, index=False)
        print(f"Saved SEC results to: {sec_output_file}")

    sec_agreement = find_agreement_percentage(sec_test_df['Pred_Stance'], sec_test_df[stance_column])
    run_summary.append({
        'Model': model_name, 'Target': target, 'Data': 'SEC',
        'Accuracy': sec_agreement, 'run_id': run_id, 'mode': mode, 'cot': use_cot
    })

    print(f"\n{'='*70}")
    print(f"Results: ECT Accuracy: {ect_agreement}% | SEC Accuracy: {sec_agreement}%")
    print(f"{'='*70}")

    return run_summary


# ============================================================================
# ARGUMENT PARSER SETUP
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run zero-shot stance classification experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
# Run zero-shot without CoT
python 1_zero-shot-experiment.py --mode no-transcript

# Run zero-shot with CoT
python 1_zero-shot-experiment.py --mode no-transcript --use-cot

# Run with summarized context, 5 runs
python 1_zero-shot-experiment.py --mode summarized --num-runs 5

# Run a specific model and targets
python 1_zero-shot-experiment.py --mode full --model llama3-sdsc --targets debt sales

# Run against an OpenAI-compatible custom endpoint
python 1_zero-shot-experiment.py --mode no-transcript --model llama3 \
    --client-base-url http://localhost:8005/v1 --client-api-key not-needed
        """
    )

    # Required arguments
    parser.add_argument(
        '--mode', type=str, required=True,
        choices=['no-transcript', 'summarized', 'full'],
        help='Context mode: no-transcript, summarized, or full'
    )

    # Optional arguments
    parser.add_argument(
        '--use-cot', action='store_true',
        help='Enable Chain-of-Thought (CoT) prompting (default: off)'
    )

    parser.add_argument(
        '--num-runs', type=int, default=3,
        help='Number of runs for each experiment (default: 3)'
    )

    parser.add_argument(
        '--model', default='gpt-4.1-mini-2025-04-14',
        help='Model name served by the configured client '
             '(default: gpt-4.1-mini-2025-04-14)'
    )

    parser.add_argument(
        '--client-api-key', '--api-key', dest='client_api_key', default=None,
        help='API key for the client (default: use OPENAI_API_KEY)'
    )

    parser.add_argument(
        '--client-base-url', '--base-url', dest='client_base_url', default=None,
        help='OpenAI-compatible API base URL (default: standard OpenAI endpoint)'
    )

    parser.add_argument(
        '--extract-model', default=None,
        help='Model used to extract the final stance. It must be served by the '
             'same client (default: use each experiment model)'
    )

    parser.add_argument(
        '--targets', type=str, nargs='+', default=TARGETS,
        choices=TARGETS,
        help=f'Targets to classify (default: {", ".join(TARGETS)})'
    )

    parser.add_argument(
        '--results-dir', type=str, default=None,
        help='Custom results directory (default: auto-generated based on mode)'
    )

    parser.add_argument(
        '--data-dir', type=str, default=DEFAULT_DATA_DIR,
        help=f'Base directory for the dataset (default: {DEFAULT_DATA_DIR})'
    )

    parser.add_argument(
        '--stance-column', type=str, default='LLM_Stance_1',
        help='Column name for ground truth stance labels (default: LLM_Stance_1)'
    )

    parser.add_argument(
        '--num-workers', type=int, default=1,
        help='Number of parallel workers for API calls (default: 1)'
    )

    parser.add_argument(
        '--yes', action='store_true',
        help='Skip confirmation prompt (useful for bash scripts)'
    )

    return parser.parse_args()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run the experiments."""
    args = parse_arguments()

    # Determine results directory
    if args.results_dir is None:
        cot_suffix = '_cot' if args.use_cot else ''
        mode_to_dir = {
            'no-transcript': f'no_context{cot_suffix}',
            'summarized': f'summarized_context{cot_suffix}',
            'full': f'full_context{cot_suffix}'
        }
        results_dir = os.path.join(DEFAULT_RESULTS_DIR, mode_to_dir[args.mode])
    else:
        results_dir = args.results_dir

    os.makedirs(results_dir, exist_ok=True)

    # Print experiment configuration
    print("\n" + "="*70)
    print("ZERO-SHOT EXPERIMENT CONFIGURATION")
    print("="*70)
    print(f"Mode:           {args.mode}")
    print(f"Chain-of-Thought: {'Yes' if args.use_cot else 'No'}")
    print(f"Model:          {args.model}")
    print(f"Client URL:     {args.client_base_url or 'OpenAI default'}")
    print(f"Extract Model:  {args.extract_model or 'same as experiment model'}")
    print(f"Targets:        {', '.join(args.targets)}")
    print(f"Num Runs:       {args.num_runs}")
    print(f"Num Workers:    {args.num_workers}")
    print(f"Data Dir:       {args.data_dir}")
    print(f"Results Dir:    {results_dir}")
    print(f"Stance Column:  {args.stance_column}")
    print("="*70 + "\n")

    # Ask for confirmation (skip with --yes)
    if not args.yes:
        response = input("Do you want to proceed with this configuration? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Experiment cancelled.")
            return

    # Load existing summary results if resuming
    # Use per-model summary file to avoid race conditions when models run in parallel
    summary_file = os.path.join(results_dir, f'all_results_summary_{args.model}.csv')
    experiment_summary = []
    if os.path.exists(summary_file):
        existing_df = pd.read_csv(summary_file)
        experiment_summary = existing_df.to_dict(orient='records')
        print(f"Loaded {len(experiment_summary)} existing results from previous run(s)")

    # Build list of experiments, skipping only fully-completed ones (both ECT and SEC done).
    # Partially-completed experiments (e.g. ECT done, SEC missing) are still queued —
    # run_experiment() will skip the finished dataset and only run the missing one.
    experiments_to_run = []
    skipped = 0
    partial = 0
    model_name = args.model
    for run_id in range(1, args.num_runs + 1):
        for target in args.targets:
            ect_output_dir = os.path.join(results_dir, model_name, 'ECT')
            sec_output_dir = os.path.join(results_dir, model_name, 'SEC')
            ect_output_file = os.path.join(ect_output_dir, f'test-ECT-{target}-{model_name}-run_{run_id}.csv')
            sec_output_file = os.path.join(sec_output_dir, f'test-SEC-{target}-{model_name}-run_{run_id}.csv')

            ect_done = os.path.exists(ect_output_file)
            sec_done = os.path.exists(sec_output_file)

            if ect_done and sec_done:
                print(f"Skipping (fully completed): {model_name}-{target}-run_{run_id}")
                skipped += 1
                continue

            if ect_done or sec_done:
                done_part = "ECT" if ect_done else "SEC"
                todo_part = "SEC" if ect_done else "ECT"
                print(f"Resuming ({done_part} done, {todo_part} pending): {model_name}-{target}-run_{run_id}")
                partial += 1

            experiments_to_run.append((model_name, target, run_id))

    total_experiments = len(experiments_to_run)
    if skipped > 0 or partial > 0:
        print(f"\nSkipped {skipped} fully-completed | Resuming {partial} partially-completed")

    if total_experiments == 0:
        print("All experiments already completed!")
        return

    client = create_client(args.client_api_key, args.client_base_url)
    print(f"Running {total_experiments} experiment(s)...\n")

    # Run experiments
    for i, (model_name, target, run_id) in enumerate(experiments_to_run, 1):
        print(f"\n\nProgress: Experiment {i}/{total_experiments}")

        run_summary = run_experiment(
            model_name, target, run_id, args.mode,
            args.use_cot, results_dir, args.data_dir, args.stance_column,
            args.num_workers, client, args.extract_model
        )
        experiment_summary.extend(run_summary)

        # Save cumulative results after each experiment (progress is never lost)
        result_df = pd.DataFrame(experiment_summary)
        result_df.to_csv(summary_file, index=False)
        print(f"Cumulative results saved to: {summary_file}")

    # Final summary
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*70)
    print(f"Experiments run: {total_experiments} | Skipped: {skipped}")
    print(f"Results saved in: {results_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
