import argparse
import os
import sys

from src.loader import load_dataset_files
from src.analyzer import DatasetAnalyzer
from src.splitter import DatasetSplitter
from src.report_generator import generate_markdown_report, generate_split_report
from src.preprocessing import run_sanity_check
from src.train_baseline import train_baseline_pipeline
from src.train_model import run_training_pipeline
from src.generate_comparison import generate_comparison


def run_dataset_analysis(data_dir: str = "dataset"):
    print("\n================================================================================")
    print(" STAGE 1: DATASET ANALYSIS")
    print("================================================================================")

    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found.")
        sys.exit(1)

    print(f"[1/3] Loading dataset files from '{data_dir}/'...")
    individual_dfs, combined_df = load_dataset_files(data_dir=data_dir)

    print(f"      Successfully loaded {len(individual_dfs)} TSV files.")
    print(f"      Combined dataset shape: {combined_df.shape[0]:,} rows x {combined_df.shape[1]} columns.")

    print("\n[2/3] Performing dataset analysis...")
    analyzer = DatasetAnalyzer(individual_dfs, combined_df)
    metrics = analyzer.run_all()

    print("\n--------------------------------------------------------------------------------")
    print(" DATASET ANALYSIS SUMMARY")
    print("--------------------------------------------------------------------------------")
    print(f"Total Combined Samples : {metrics['shapes']['combined'][0]:,}")
    print(f"Columns                : {metrics['columns']['columns']}")
    print(f"Total Missing Values   : {metrics['missing_values']['total_missing']}")
    print("\nClass Distribution (Combined):")
    comb_dist = metrics['class_distribution']['combined']
    for label, count in comb_dist['counts'].items():
        pct = comb_dist['percentages'].get(label, 0.0)
        print(f"  - {label:<12}: {count:>6,} ({pct:6.2f}%)")

    print("\nDuplicates & Consistency Audit:")
    print(f"  - Raw Exact Duplicate Rows         : {metrics['exact_duplicates']['raw_content_duplicates']}")
    print(f"  - Full Duplicate Rows (w/ Source)  : {metrics['exact_duplicates']['full_row_duplicates']}")
    print(f"  - Duplicate Sentence Pairs         : {metrics['duplicate_sentence_pairs']['duplicate_pairs_count']}")
    print(f"  - Conflicting Label Pairs Count    : {metrics['conflicting_labels']['conflicting_pairs_count']}")

    print("\n[3/3] Saving dataset analysis report...")
    output_report_path = os.path.join("results", "dataset_analysis_report.md")
    generate_markdown_report(metrics, output_path=output_report_path)
    print(f"      Report written to: {output_report_path}")
    print("================================================================================\n")
    return individual_dfs, combined_df


def run_dataset_split(combined_df=None, data_dir: str = "dataset"):
    print("\n================================================================================")
    print(" STAGE 2: EXPERIMENTAL TRAIN / VAL / TEST SPLIT (GROUP-AWARE)")
    print("================================================================================")

    if combined_df is None:
        print(f"[1/4] Loading combined dataset from '{data_dir}/'...")
        _, combined_df = load_dataset_files(data_dir=data_dir)

    splitter = DatasetSplitter(combined_df, random_state=42)

    print("\n[2/4] Pre-split check for duplicate sentence pairs...")
    pre_dup_info = splitter.check_pre_split_duplicates()
    print(f"      Total rows               : {pre_dup_info['total_rows']:,}")
    print(f"      Unique sentence pairs    : {pre_dup_info['unique_sentence_pairs']:,}")
    print(f"      Duplicate sentence pairs : {pre_dup_info['duplicate_pairs_count']}")

    print("\n[3/4] Performing group-aware 80/10/10 stratified split (random_state=42)...")
    train_df, val_df, test_df = splitter.split_data(train_ratio=0.80, val_ratio=0.10, test_ratio=0.10)
    saved_paths = splitter.save_splits(output_dir=data_dir)

    print(f"      Train split      : {len(train_df):>6,} rows -> {saved_paths['train']}")
    print(f"      Validation split : {len(val_df):>6,} rows -> {saved_paths['validation']}")
    print(f"      Test split       : {len(test_df):>6,} rows -> {saved_paths['test']}")

    print("\n[4/4] Verifying split partitions & generating report...")
    split_metrics = splitter.verify_splits()

    print("\n--------------------------------------------------------------------------------")
    print(" SPLIT VERIFICATION SUMMARY")
    print("--------------------------------------------------------------------------------")
    print("Row Counts per Split:")
    for key, count in split_metrics["row_counts"].items():
        print(f"  - {key.capitalize():<12}: {count:>6,} rows (Missing: {split_metrics['missing_values'][key]})")

    print("\nClass Distributions:")
    for key, dist in split_metrics["class_distributions"].items():
        counts_str = ", ".join([f"{l}: {c}" for l, c in dist["counts"].items()])
        print(f"  - {key.capitalize():<12}: {counts_str}")

    print("\nIntra-Split Duplicate Sentence Pairs:")
    for key, idup in split_metrics["internal_duplicates"].items():
        print(f"  - {key.capitalize():<12}: {idup} internal duplicates")

    print("\nInter-Split Sentence Pair Overlap Check:")
    ov = split_metrics["overlap_analysis"]
    print(f"  - Train vs Val Overlap  : {ov['train_val_overlap_count']} pairs")
    print(f"  - Train vs Test Overlap : {ov['train_test_overlap_count']} pairs")
    print(f"  - Val vs Test Overlap   : {ov['val_test_overlap_count']} pairs")

    output_report_path = os.path.join("results", "split_analysis_report.md")
    generate_split_report(split_metrics, output_path=output_report_path)
    print(f"\n      Split report written to: {output_report_path}")
    print("================================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Enhanced Temporal Fact Verification Pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        default="sanity",
        choices=["analysis", "split", "sanity", "baseline", "model", "comparison"],
        help="Pipeline stage to execute."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bert",
        help="Model type for training ('bert', 'roberta', 'distilbert')."
    )
    parser.add_argument(
        "--sanity_check",
        action="store_true",
        help="Run quick sanity check for training pipeline."
    )
    args = parser.parse_args()

    if args.stage == "analysis":
        run_dataset_analysis()
    elif args.stage == "split":
        run_dataset_split()
    elif args.stage == "sanity":
        run_sanity_check()
    elif args.stage == "baseline":
        train_baseline_pipeline()
    elif args.stage == "model":
        run_training_pipeline(model_arg=args.model, sanity_check=args.sanity_check)
    elif args.stage == "comparison":
        generate_comparison()


if __name__ == "__main__":
    main()
