"""
Training pipeline for the MedAI chest X-ray classifier.

Expected dataset structure:

    <data_dir>/
        train/
            NORMAL/
            PNEUMONIA/
        val/
            NORMAL/
            PNEUMONIA/
        test/
            NORMAL/
            PNEUMONIA/

This script trains the SAME MedicalCNN architecture used by
src/ml/cnn_model.py.

The model is intended for academic/demo use only and is NOT
clinically validated.
"""

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def build_transforms():
    """Build training and evaluation transforms."""
    from torchvision import transforms

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_tf, eval_tf


def evaluate(model, loader, device):
    """
    Evaluate model and return classification metrics.

    Returns:
        accuracy, precision, recall, f1, confusion_matrix
    """
    import torch

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    if not all_labels:
        return 0.0, 0.0, 0.0, 0.0, [[0, 0], [0, 0]]

    # Calculate metrics without requiring extra packages.
    tp = sum(
        1 for p, y in zip(all_predictions, all_labels)
        if p == 1 and y == 1
    )
    tn = sum(
        1 for p, y in zip(all_predictions, all_labels)
        if p == 0 and y == 0
    )
    fp = sum(
        1 for p, y in zip(all_predictions, all_labels)
        if p == 1 and y == 0
    )
    fn = sum(
        1 for p, y in zip(all_predictions, all_labels)
        if p == 0 and y == 1
    )

    total = len(all_labels)

    accuracy = (tp + tn) / total if total else 0.0

    precision = tp / (tp + fp) if (tp + fp) else 0.0

    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    confusion_matrix = [
        [tn, fp],
        [fn, tp]
    ]

    return accuracy, precision, recall, f1, confusion_matrix


def train(
    data_dir: str,
    epochs: int = 8,
    batch_size: int = 16,
    lr: float = 1e-3
):
    """
    Train MedicalCNN on a real ImageFolder dataset.
    """

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    from src.ml.cnn_model import MedicalCNN

    data_path = Path(data_dir)

    train_dir = data_path / "train"
    val_dir = data_path / "val"
    test_dir = data_path / "test"

    # ---------------------------------------------------------
    # Check dataset
    # ---------------------------------------------------------

    if not train_dir.exists() or not any(train_dir.iterdir()):
        raise FileNotFoundError(
            f"No training data found at {train_dir}."
        )

    train_tf, eval_tf = build_transforms()

    train_ds = ImageFolder(
        str(train_dir),
        transform=train_tf
    )

    if len(train_ds.classes) != 2:
        raise ValueError(
            "Expected exactly two classes (NORMAL, PNEUMONIA), "
            f"found: {train_ds.classes}"
        )

    logger.info(
        f"Classes detected: {train_ds.classes}"
    )

    logger.info(
        f"Training images: {len(train_ds)}"
    )

    # ---------------------------------------------------------
    # Class imbalance handling
    # ---------------------------------------------------------

    # ImageFolder maps classes alphabetically:
    # NORMAL -> 0
    # PNEUMONIA -> 1

    class_counts = [0] * len(train_ds.classes)

    for _, label in train_ds.samples:
        class_counts[label] += 1

    logger.info(
        f"Class distribution: {dict(zip(train_ds.classes, class_counts))}"
    )

    total_samples = sum(class_counts)

    class_weights = [
        total_samples / (len(class_counts) * count)
        for count in class_counts
    ]

    logger.info(
        f"Class weights: {class_weights}"
    )

    # ---------------------------------------------------------
    # Data loaders
    # ---------------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = None

    if val_dir.exists() and any(val_dir.iterdir()):
        val_ds = ImageFolder(
            str(val_dir),
            transform=eval_tf
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        logger.info(
            f"Validation images: {len(val_ds)}"
        )

    test_loader = None

    if test_dir.exists() and any(test_dir.iterdir()):
        test_ds = ImageFolder(
            str(test_dir),
            transform=eval_tf
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        logger.info(
            f"Test images: {len(test_ds)}"
        )

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info(
        f"Training device: {device}"
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = MedicalCNN(
        num_classes=2
    ).to(device)

    # Weighted loss handles the NORMAL/PNEUMONIA imbalance.
    weight_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=device
    )

    criterion = nn.CrossEntropyLoss(
        weight=weight_tensor
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    logger.info(
        "Starting training..."
    )

    best_val_accuracy = -1.0
    best_state = None

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = (
            running_loss / total
            if total
            else 0.0
        )

        train_accuracy = (
            correct / total
            if total
            else 0.0
        )

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f}"
        )

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        if val_loader is not None:

            val_accuracy, val_precision, val_recall, val_f1, _ = (
                evaluate(
                    model,
                    val_loader,
                    device
                )
            )

            logger.info(
                f"             "
                f"val_acc={val_accuracy:.4f} | "
                f"val_precision={val_precision:.4f} | "
                f"val_recall={val_recall:.4f} | "
                f"val_f1={val_f1:.4f}"
            )

            # Save the best model according to validation accuracy.
            if val_accuracy > best_val_accuracy:

                best_val_accuracy = val_accuracy

                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    # ---------------------------------------------------------
    # Restore best model
    # ---------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(best_state)

        logger.info(
            f"Restored best validation model "
            f"(val_acc={best_val_accuracy:.4f})"
        )

    # ---------------------------------------------------------
    # Test evaluation
    # ---------------------------------------------------------

    if test_loader is not None:

        test_accuracy, test_precision, test_recall, test_f1, cm = (
            evaluate(
                model,
                test_loader,
                device
            )
        )

        logger.info("")
        logger.info("=" * 60)
        logger.info("FINAL TEST RESULTS")
        logger.info("=" * 60)

        logger.info(
            f"Test Accuracy : {test_accuracy:.4f}"
        )

        logger.info(
            f"Test Precision: {test_precision:.4f}"
        )

        logger.info(
            f"Test Recall   : {test_recall:.4f}"
        )

        logger.info(
            f"Test F1 Score : {test_f1:.4f}"
        )

        logger.info(
            f"Confusion Matrix: {cm}"
        )

        logger.info("=" * 60)

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    model_path = MODEL_DIR / "xray_cnn.pth"

    torch.save(
        model.state_dict(),
        model_path
    )

    logger.info(
        f"Saved trained weights to {model_path}"
    )

    logger.info(
        "IMPORTANT: This is an academic/demo model and is "
        "NOT clinically validated. Test results must not be "
        "presented as clinical performance."
    )


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Train the MedAI chest X-ray classifier "
            "on a real dataset."
        )
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help=(
            "Path to the chest X-ray dataset "
            "containing train/, val/, and test/"
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=8
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    args = parser.parse_args()

    train(
        args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )