# ClearML Research Guide

This project is integrated with **ClearML** to make your research reproducible and organized. This guide explains how to leverage the experimental features effectively.

## 1. Experimental Workflow

The core workflow for conducting research is:
1.  **Hypothesize**: Decide what you want to test (e.g., "Does increasing dropout to 0.3 improve generalization?").
2.  **Run**: Execute the training command with specific arguments and a descriptive comment.
3.  **Compare**: Use the ClearML UI to compare the new run against a baseline.

### Command Structure
Always use descriptive comments!

```bash
pixi run python main.py --data tinystories --dropout 0.3 --comment "Hypothesis: Increased dropout improves val loss"
```

## 2. Organizing Experiments (Tags)

The system now **automatically tags** your experiments based on the dataset used:
- `tinystories`: Untagged validation set runs.
- `filename.txt`: Custom local datasets.

You can also add custom tags in the Web UI to group experiments (e.g., `baseline`, `experimental`, `production`).

## 3. How to Compare Runs

1.  Go to the **ClearML Web UI**.
2.  Select your project (`gpt-from-scratch`).
3.  **Select Multiple Experiments**: Click the checkboxes next to the runs you want to compare (e.g., your baseline vs. your new run).
4.  Click the **Inspect / Compare** button (bottom bar or context menu).

### Key Metrics to Watch
- **Scalars -> Loss/val**: This is your primary metric. Lower is better. Look for the crossover point where the new model beats the old one.
- **Scalars -> Loss/train**: Check this to detect overfitting (if train drops hard but val stays high).
- **Text -> Generated Sample**: Qualitative check. Does the text make sense?

## 4. Reproducing Results

ClearML captures *everything* needed to reproduce a run:
- **Git Commit**: The exact code version.
- **Uncommitted Changes**: Even your uncommitted edits are saved as a patch.
- **Arguments**: All command-line args (`--learning_rate`, etc.) are saved in `Configuration > Hyperparameters`.

To reproduce a specific experiment locally:
1.  Checkout the code state (or just view the "Configuration" tab in UI to see what args were used).
2.  Run the command with those exact arguments.

## 5. Tips for "Research Mode"

- **Change One Thing at a Time**: Don't change LR and Dropout and Layers all at once. You won't know what caused the improvement.
- **Use the `--comment` flag**: It saves you from guessing "what changed in this run?" a week later.
- **Abort Early**: If the loss curve is clearly exploding or worse than baseline after 20% of steps, kill it (Ctrl+C). Don't waste compute.
