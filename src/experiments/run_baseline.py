from baseline_runs import BaselineExperiment

if __name__ == "__main__":
    experiment = BaselineExperiment(n_runs=10)
    results = experiment.run_experiment("data/creditcard.csv")

    # Print aggregate results
    print("\nAggregate Results:")
    print("-" * 40)
    metrics_of_interest = ['accuracy', 'recision', 'recall', 'f1_score', 'roc_auc']
    for metrics in metrics_of_interest:
        print(f"\n{metic.upper()}:")
        print(f"Mean: {results[f'{metric}_mean']:.4f}")
        print(f"Std Dev: {results[f'{metric}_std']:.4f}")
        print(f"95% CI: [{results[f'{metric}_ci_lower']:.4f}, {results[f'{metric}_ci_upper']:.4f}]")