import torch


def _compute_roc(
    y_pred: torch.Tensor, y_true: torch.Tensor, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # From 0 = genuine, 1 = morphed,
    # we want 0 = morphed (negative), 1 = genuine (positive)
    y_pred_n = 1 - y_pred
    y_true_n = 1 - y_true
    # Initialize the number of predictions, the number of positives and negatives
    num_preds = y_pred.shape[0]
    num_positives = torch.count_nonzero(y_true_n)
    num_negatives = num_preds - num_positives
    if num_positives == 0:
        raise ValueError("No positive samples in y_true")
    if num_negatives == 0:
        raise ValueError("No negative samples in y_true")
    predicted_positives = torch.arange(1, num_preds + 1, device=device)
    # Sort the predictions by their confidence, then by their descending label
    _, sorted_true = torch.sort(y_true_n, stable=True, descending=True)
    _, sorted_pred = torch.sort(y_pred_n[sorted_true], stable=True)
    # Count false positives and false negatives
    false_positives = torch.cumsum(y_true_n[sorted_true][sorted_pred], dim=0)
    # HACK: Why num_negatives?
    false_negatives = num_negatives - (predicted_positives - false_positives)
    # Compute Pfa and Pmiss
    Pfa = torch.zeros(num_preds + 1, device=device)  # noqa: N806
    Pmiss = torch.zeros(num_preds + 1, device=device)  # noqa: N806
    Pfa[0] = 1
    Pmiss[0] = 0
    Pfa[1:] = false_negatives / num_negatives
    Pmiss[1:] = false_positives / num_positives
    return Pfa, Pmiss
