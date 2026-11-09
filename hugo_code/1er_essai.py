"""Train a small supervised baseline and export it as an ONNX model.

Run from this folder with the project dependencies installed:
    python 1er_essai.py --data-dir labeled --output model_baseline.onnx
"""

import argparse
import csv
from pathlib import Path
import time

import blosc2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


TRACE_LENGTH = 12_000
WINDOW_LENGTH = 600
NUM_CLASSES = 5
NUM_FEATURES = 17
EPSILON = 1e-8


def extract_numpy_features(csi: np.ndarray, ati: np.ndarray) -> np.ndarray:
    """Summarize one 600-frame complex CSI window and its timing intervals."""
    magnitude = np.log1p(np.abs(csi[:, 0, :, :])).astype(np.float32, copy=False)

    channel_mean = magnitude.mean(axis=(0, 2))
    centered = magnitude - channel_mean[None, :, None]
    channel_std = np.sqrt(np.mean(centered**2, axis=(0, 2)) + EPSILON)

    frame_mean = magnitude.mean(axis=2)
    temporal_std = np.sqrt(
        np.mean((frame_mean - frame_mean.mean(axis=0)) ** 2, axis=0) + EPSILON
    )
    temporal_delta = np.abs(np.diff(frame_mean, axis=0)).mean(axis=0)

    frequency_centered = magnitude - magnitude.mean(axis=2, keepdims=True)
    frequency_std = np.sqrt(
        np.mean(frequency_centered**2, axis=2) + EPSILON
    ).mean(axis=0)

    log_intervals = np.log1p(np.maximum(ati, 0))
    interval_mean = log_intervals.mean()
    interval_std = np.sqrt(
        np.mean((log_intervals - interval_mean) ** 2) + EPSILON
    )

    features = np.concatenate(
        (
            channel_mean,
            channel_std,
            temporal_std,
            temporal_delta,
            frequency_std,
            np.asarray([interval_mean, interval_std], dtype=np.float32),
        )
    )
    return features.astype(np.float32, copy=False)


def load_training_windows(data_dir: Path):
    feature_rows = []
    labels = []
    example_frames = None
    example_intervals = None

    with (data_dir / "samples.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No training rows found in {data_dir / 'samples.csv'}")

    for row in rows:
        identifier = row["identifier"]
        try:
            label = int(row["headcount"])
        except ValueError as exc:
            raise ValueError(
                f"Expected an integer headcount for labeled sample {identifier!r}"
            ) from exc
        if not 0 <= label < NUM_CLASSES:
            raise ValueError(f"Headcount {label} for {identifier!r} is outside 0-4")

        csi_path = data_dir / f"CSI_{identifier}.b2"
        ati_path = data_dir / f"ATI_{identifier}.b2"
        if not csi_path.is_file() or not ati_path.is_file():
            raise FileNotFoundError(f"Missing CSI or ATI data for {identifier!r}")

        csi = blosc2.load_array(str(csi_path))
        ati = blosc2.load_array(str(ati_path))
        if csi.shape != (TRACE_LENGTH, 1, 3, 1024) or csi.dtype != np.complex64:
            raise ValueError(
                f"Unexpected CSI format for {identifier!r}: "
                f"shape={csi.shape}, dtype={csi.dtype}"
            )
        if ati.shape != (TRACE_LENGTH,) or ati.dtype != np.float32:
            raise ValueError(
                f"Unexpected ATI format for {identifier!r}: "
                f"shape={ati.shape}, dtype={ati.dtype}"
            )
        if not np.isfinite(ati).all() or np.any(ati < 0):
            raise ValueError(f"ATI contains invalid values for {identifier!r}")

        for start in range(0, TRACE_LENGTH, WINDOW_LENGTH):
            end = start + WINDOW_LENGTH
            features = extract_numpy_features(csi[start:end], ati[start:end])
            if not np.isfinite(features).all():
                raise ValueError(f"Extracted features are invalid for {identifier!r}")
            feature_rows.append(features)
            labels.append(label)

            if example_frames is None:
                window = csi[start:end]
                example_frames = np.stack(
                    (window.real, window.imag), axis=-1
                ).astype(np.float32, copy=False)
                example_intervals = ati[start:end].copy()
                example_intervals[-1] = 0

    if example_frames is None or example_intervals is None:
        raise ValueError("No labeled CSI windows were loaded")

    return (
        np.stack(feature_rows),
        np.asarray(labels, dtype=np.int64),
        example_frames,
        example_intervals,
    )


def train_model(
    features: np.ndarray, labels: np.ndarray, epochs: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_mean = features.mean(axis=0).astype(np.float32)
    feature_scale = features.std(axis=0).astype(np.float32)
    feature_scale = np.maximum(feature_scale, 1e-6)
    normalized = (features - feature_mean) / feature_scale

    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    class_weights = np.zeros(NUM_CLASSES, dtype=np.float32)
    present = class_counts > 0
    class_weights[present] = len(labels) / (
        int(present.sum()) * class_counts[present]
    )
    sample_weights = class_weights[labels]
    sample_weights /= sample_weights.sum()

    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 0.01, (NUM_CLASSES, NUM_FEATURES)).astype(np.float32)
    bias = np.zeros(NUM_CLASSES, dtype=np.float32)
    first_moment_w = np.zeros_like(weights)
    second_moment_w = np.zeros_like(weights)
    first_moment_b = np.zeros_like(bias)
    second_moment_b = np.zeros_like(bias)
    beta1, beta2 = 0.9, 0.999
    learning_rate, weight_decay = 0.03, 1e-3

    for epoch in range(epochs):
        logits = normalized @ weights.T + bias
        shifted_logits = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted_logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        loss = -np.sum(
            sample_weights * np.log(probabilities[np.arange(len(labels)), labels] + 1e-12)
        )

        gradients = probabilities
        gradients[np.arange(len(labels)), labels] -= 1
        gradients *= sample_weights[:, None]
        gradient_w = gradients.T @ normalized
        gradient_b = gradients.sum(axis=0)

        step = epoch + 1
        first_moment_w = beta1 * first_moment_w + (1 - beta1) * gradient_w
        second_moment_w = beta2 * second_moment_w + (1 - beta2) * gradient_w**2
        first_moment_b = beta1 * first_moment_b + (1 - beta1) * gradient_b
        second_moment_b = beta2 * second_moment_b + (1 - beta2) * gradient_b**2
        corrected_w = first_moment_w / (1 - beta1**step)
        corrected_w /= np.sqrt(second_moment_w / (1 - beta2**step)) + 1e-8
        corrected_b = first_moment_b / (1 - beta1**step)
        corrected_b /= np.sqrt(second_moment_b / (1 - beta2**step)) + 1e-8
        weights *= 1 - learning_rate * weight_decay
        weights -= learning_rate * corrected_w
        bias -= learning_rate * corrected_b

        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch + 1 == epochs:
            accuracy = np.mean(logits.argmax(axis=1) == labels)
            print(
                f"Epoch {epoch + 1:>3}/{epochs}: "
                f"weighted loss={float(loss):.4f}, training window accuracy={accuracy:.1%}"
            )

    return weights, bias, feature_mean, feature_scale


def export_and_check(
    weights: np.ndarray,
    bias: np.ndarray,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    frames: np.ndarray,
    intervals: np.ndarray,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    nodes = []
    initializers = []

    def constant(name: str, value: np.ndarray) -> str:
        initializers.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    def node(op_type: str, inputs: list[str], output_name: str, **attributes) -> str:
        nodes.append(helper.make_node(op_type, inputs, [output_name], **attributes))
        return output_name

    def reduce_mean(
        input_name: str, axes: list[int], output_name: str, keepdims: int = 0
    ) -> str:
        axes_name = constant(
            f"{output_name}_axes", np.asarray(axes, dtype=np.int64)
        )
        return node(
            "ReduceMean",
            [input_name, axes_name],
            output_name,
            keepdims=keepdims,
        )

    real = node(
        "Gather",
        ["frames", constant("real_component_index", np.asarray(0, dtype=np.int64))],
        "real",
        axis=4,
    )
    imaginary = node(
        "Gather",
        [
            "frames",
            constant("imag_component_index", np.asarray(1, dtype=np.int64)),
        ],
        "imaginary",
        axis=4,
    )
    real_squared = node("Mul", [real, real], "real_squared")
    imaginary_squared = node("Mul", [imaginary, imaginary], "imaginary_squared")
    magnitude_squared = node(
        "Add", [real_squared, imaginary_squared], "magnitude_squared"
    )
    stabilized_magnitude = node(
        "Add",
        [magnitude_squared, constant("epsilon", np.asarray(EPSILON, np.float32))],
        "stabilized_magnitude",
    )
    magnitude = node("Sqrt", [stabilized_magnitude], "magnitude_with_tx")
    magnitude = node(
        "Squeeze",
        [magnitude, constant("squeeze_tx_axis", np.asarray([1], dtype=np.int64))],
        "magnitude",
    )
    magnitude = node(
        "Log",
        [
            node(
                "Add",
                [magnitude, constant("one", np.asarray(1, dtype=np.float32))],
                "magnitude_plus_one",
            )
        ],
        "log_magnitude",
    )

    channel_mean = reduce_mean(magnitude, [0, 2], "channel_mean")
    centered_channel = node(
        "Sub",
        [
            magnitude,
            node(
                "Unsqueeze",
                [
                    channel_mean,
                    constant("channel_broadcast_axes", np.asarray([0, 2], np.int64)),
                ],
                "channel_mean_broadcast",
            ),
        ],
        "centered_channel",
    )
    channel_variance = reduce_mean(
        node("Mul", [centered_channel, centered_channel], "centered_channel_squared"),
        [0, 2],
        "channel_variance",
    )
    channel_std = node(
        "Sqrt",
        [
            node(
                "Add",
                [
                    channel_variance,
                    constant("channel_variance_epsilon", np.asarray(EPSILON, np.float32)),
                ],
                "stabilized_channel_variance",
            )
        ],
        "channel_std",
    )

    frame_mean = reduce_mean(magnitude, [2], "frame_mean")
    temporal_mean = reduce_mean(frame_mean, [0], "temporal_mean")
    centered_temporal = node(
        "Sub",
        [
            frame_mean,
            node(
                "Unsqueeze",
                [
                    temporal_mean,
                    constant("temporal_broadcast_axes", np.asarray([0], np.int64)),
                ],
                "temporal_mean_broadcast",
            ),
        ],
        "centered_temporal",
    )
    temporal_variance = reduce_mean(
        node("Mul", [centered_temporal, centered_temporal], "centered_temporal_squared"),
        [0],
        "temporal_variance",
    )
    temporal_std = node(
        "Sqrt",
        [
            node(
                "Add",
                [
                    temporal_variance,
                    constant("temporal_variance_epsilon", np.asarray(EPSILON, np.float32)),
                ],
                "stabilized_temporal_variance",
            )
        ],
        "temporal_std",
    )

    before_last_frame = node(
        "Slice",
        [
            frame_mean,
            constant("slice_start", np.asarray([0], np.int64)),
            constant("slice_end", np.asarray([599], np.int64)),
            constant("slice_axis", np.asarray([0], np.int64)),
        ],
        "frame_mean_before_last",
    )
    after_first_frame = node(
        "Slice",
        [
            frame_mean,
            constant("slice_next_start", np.asarray([1], np.int64)),
            constant("slice_next_end", np.asarray([600], np.int64)),
            constant("slice_next_axis", np.asarray([0], np.int64)),
        ],
        "frame_mean_after_first",
    )
    frame_delta = node(
        "Abs",
        [node("Sub", [after_first_frame, before_last_frame], "frame_difference")],
        "absolute_frame_difference",
    )
    temporal_delta = reduce_mean(frame_delta, [0], "temporal_delta")

    frequency_mean = reduce_mean(magnitude, [2], "frequency_mean", keepdims=1)
    centered_frequency = node(
        "Sub", [magnitude, frequency_mean], "centered_frequency"
    )
    frequency_variance = reduce_mean(
        node("Mul", [centered_frequency, centered_frequency], "centered_frequency_squared"),
        [2],
        "frequency_variance",
    )
    frequency_std_per_frame = node(
        "Sqrt",
        [
            node(
                "Add",
                [
                    frequency_variance,
                    constant("frequency_variance_epsilon", np.asarray(EPSILON, np.float32)),
                ],
                "stabilized_frequency_variance",
            )
        ],
        "frequency_std_per_frame",
    )
    frequency_std = reduce_mean(
        frequency_std_per_frame, [0], "frequency_std"
    )

    clipped_intervals = node(
        "Clip",
        [
            "intervals",
            constant("interval_min", np.asarray(0, np.float32)),
            constant("interval_max", np.asarray(1e30, np.float32)),
        ],
        "clipped_intervals",
    )
    log_intervals = node(
        "Log",
        [
            node(
                "Add",
                [
                    clipped_intervals,
                    constant("interval_one", np.asarray(1, np.float32)),
                ],
                "intervals_plus_one",
            )
        ],
        "log_intervals",
    )
    interval_mean = reduce_mean(log_intervals, [0], "interval_mean")
    centered_intervals = node(
        "Sub", [log_intervals, interval_mean], "centered_intervals"
    )
    interval_variance = reduce_mean(
        node("Mul", [centered_intervals, centered_intervals], "centered_intervals_squared"),
        [0],
        "interval_variance",
    )
    interval_std = node(
        "Sqrt",
        [
            node(
                "Add",
                [
                    interval_variance,
                    constant("interval_variance_epsilon", np.asarray(EPSILON, np.float32)),
                ],
                "stabilized_interval_variance",
            )
        ],
        "interval_std",
    )
    interval_features = node(
        "Concat",
        [
            node(
                "Unsqueeze",
                [
                    interval_mean,
                    constant("interval_mean_axis", np.asarray([0], np.int64)),
                ],
                "interval_mean_vector",
            ),
            node(
                "Unsqueeze",
                [
                    interval_std,
                    constant("interval_std_axis", np.asarray([0], np.int64)),
                ],
                "interval_std_vector",
            ),
        ],
        "interval_features",
        axis=0,
    )
    features = node(
        "Concat",
        [
            channel_mean,
            channel_std,
            temporal_std,
            temporal_delta,
            frequency_std,
            interval_features,
        ],
        "features",
        axis=0,
    )
    normalized_features = node(
        "Div",
        [
            node(
                "Sub",
                [features, constant("feature_mean", feature_mean.astype(np.float32))],
                "centered_features",
            ),
            constant("feature_scale", feature_scale.astype(np.float32)),
        ],
        "normalized_features",
    )
    logits = node(
        "Add",
        [
            node(
                "MatMul",
                [
                    normalized_features,
                    constant("classifier_weights", weights.T.astype(np.float32)),
                ],
                "classifier_output",
            ),
            constant("classifier_bias", bias.astype(np.float32)),
        ],
        "logits",
    )

    graph = helper.make_graph(
        nodes,
        "RoomSenseBaseline",
        [
            helper.make_tensor_value_info(
                "frames", TensorProto.FLOAT, [600, 1, 3, 1024, 2]
            ),
            helper.make_tensor_value_info("intervals", TensorProto.FLOAT, [600]),
        ],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [5])],
        initializer=initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)]
    )
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save_model(model, str(output))

    size = output.stat().st_size
    if size >= 100_000_000:
        raise ValueError(f"Exported model is too large: {size:,} bytes")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(output), sess_options=options, providers=["CPUExecutionProvider"]
    )
    inputs = {"frames": frames, "intervals": intervals}
    session.run(["logits"], inputs)
    start = time.perf_counter()
    logits = session.run(["logits"], inputs)[0]
    elapsed = time.perf_counter() - start
    if logits.shape != (NUM_CLASSES,) or logits.dtype != np.float32:
        raise ValueError(
            f"Unexpected ONNX output: shape={logits.shape}, dtype={logits.dtype}"
        )
    if not np.isfinite(logits).all():
        raise ValueError("ONNX model returned non-finite logits")

    print(f"Saved {output} ({size:,} bytes)")
    print(f"Example predicted headcount: {int(logits.argmax())}")
    print(f"CPU prediction time: {elapsed:.3f} s/sample after warm-up")
    if elapsed >= 0.7:
        print("Warning: prediction exceeds the competition's 0.7 s/sample budget.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "labeled",
        help="Folder containing samples.csv and the labeled CSI/ATI files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model_baseline.onnx"),
        help="Destination ONNX model path",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if not args.data_dir.is_dir():
        parser.error(f"Training data folder not found: {args.data_dir}")

    features, labels, frames, intervals = load_training_windows(args.data_dir)
    print(
        f"Loaded {len(features)} windows from labeled traces; "
        f"class counts (windows): {np.bincount(labels, minlength=NUM_CLASSES).tolist()}"
    )
    print(
        "Training accuracy is only a sanity check; it is not a held-out "
        "estimate of generalization to unseen rooms."
    )
    weights, bias, feature_mean, feature_scale = train_model(
        features, labels, args.epochs, args.seed
    )
    export_and_check(
        weights, bias, feature_mean, feature_scale, frames, intervals, args.output
    )


if __name__ == "__main__":
    main()
