## 📁 Repository Structure

```text
T-F-V/
├── dataset/                    # Raw and split TSV dataset files
│   ├── dataset1.tsv            # Original raw chunk 1
│   ├── dataset2.tsv            # Original raw chunk 2
│   ├── dataset3.tsv            # Original raw chunk 3
│   ├── dataset4.tsv            # Original raw chunk 4
│   ├── dataset5.tsv            # Original raw chunk 5
│   ├── train.tsv               # Training split (80%, 8,527 rows)
│   ├── validation.tsv          # Validation split (10%, 1,066 rows)
│   └── test.tsv                # Test split (10%, 1,066 rows)
├── src/                        # Modular source code
│   ├── __init__.py
│   ├── loader.py               # Dataset loading and source assignment
│   ├── analyzer.py             # Dataset statistics and consistency analysis
│   ├── splitter.py             # Group-aware stratified dataset splitter
│   └── report_generator.py     # Markdown report generation
├── notebooks/                  # Interactive notebooks and research
├── results/                    # Analysis reports and evaluation metrics
│   ├── dataset_analysis_report.md
│   └── split_analysis_report.md
├── models/                     # Model configurations and checkpoints
├── main.py                     # CLI entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Git ignore rules
```

---

## 📊 Dataset Overview and Group-Aware Split Methodology

### Dataset Integrity and Provenance

* **Total samples:** 10,659 sentence pairs across five TSV files (`dataset1.tsv`–`dataset5.tsv`).
* **Unique sentence pairs:** 10,657 unique `(sentence1, sentence2)` groups.
* **Target labels:** `Support`, `Invalidate`, and `Neutral`.
* **Columns:** `gold_label`, `sentence1`, `sentence2`, and `source`.

### Group-Aware Stratified 80/10/10 Split Strategy

The splitting process in `src/splitter.py` aims to prevent sentence-pair leakage between training, validation, and test partitions.

* **Pair grouping:** Each unique `(sentence1, sentence2)` pair is treated as a group.
* **Leakage prevention:** Duplicate pairs, including pairs with conflicting labels, should remain in the same partition.
* **Training set:** `dataset/train.tsv` — 8,527 rows.
* **Validation set:** `dataset/validation.tsv` — 1,066 rows.
* **Test set:** `dataset/test.tsv` — 1,066 rows.
* **Random seed:** `random_state=42`.
* **Class balance:** Intended to preserve approximately equal representation of the three labels in each partition.

---

## 🚀 Quick Start and Installation

### 1. Set Up the Environment

Ensure Python 3.8 or later is installed, then run:

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

Run the complete analysis and splitting pipeline:

```bash
python main.py
```

Run dataset analysis only:

```bash
python main.py --stage analysis
```

Run dataset splitting only:

```bash
python main.py --stage split
```

### 3. Generated Reports

* Dataset analysis: `results/dataset_analysis_report.md`
* Split analysis: `results/split_analysis_report.md`

---

## 📝 Citation and Reference

Hosokawa, T., Jatowt, A., & Sugiyama, K. (2024). *Temporal validity reassessment: commonsense reasoning about information obsoleteness*. Discover Computing.
