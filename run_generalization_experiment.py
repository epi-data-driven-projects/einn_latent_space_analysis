"""Run the legacy SEIRM generalization experiment with the refactored EINN."""

from experiment.generalization_experiment import EINNGeneralizationExperiment


if __name__ == "__main__":
    experiment = EINNGeneralizationExperiment(
        output_dir="./experiment_results/generalization",
        target_idx=2,
        num_states=20,
        seq_len=200,
        # Window settings: 10 prefix endpoints (20, 40, ..., 200),
        # each with 20 overlap steps + 56 forecast steps.
        min_sequence_length=20,
        forecast_horizon=56,
        window_stride=20,
        calibration_epochs=300,
        calibration_lr=0.001,
        training_epochs=20,
        batch_size=32,
        seed=42,
        device='cuda:1'
    )
    result = experiment.run(force_regenerate=False)
    print(f"Experiment completed. Outputs: {result.output_dir}")
