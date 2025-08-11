#third party imports
import torch
import numpy as np

#custom imports
from cadtoseq.constants import VOCAB
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class Sequence_comparator:
    def __init__(self, vocab, max_shift=1, ignore_after_stop=True):

        self.stop_token = vocab["STOP"]
        self.pad_token = vocab["PAD"]
        self.token_names = {v: k for k, v in vocab.items()}
        self.max_shift = max_shift
        self.ignore_after_stop = ignore_after_stop

    def _create_mask(self, seq):
        valid = seq != self.stop_token

        if self.ignore_after_stop:
            masks = []
            for s in seq:
                stop_positions = (s == self.stop_token).nonzero(as_tuple=True)
                if len(stop_positions[0]) > 0:
                    stop_idx = stop_positions[0][0]
                    mask = torch.zeros_like(s, dtype=torch.bool)
                    mask[:stop_idx] = True
                else:
                    mask = torch.ones_like(s, dtype=torch.bool)
                masks.append(mask)
            return torch.stack(masks)
        else:
            return valid

    def elementwise_accuracy(self, a, b, mask):
        correct = (a == b) & mask
        seq_accuracies = []

        for i in range(a.size(0)):
            valid_positions = mask[i].sum()
            if valid_positions > 0:
                acc = correct[i].sum().float() / valid_positions.float()
            else:
                acc = torch.tensor(0.0, device=a.device)  # optional: NaN falls gewünscht
            seq_accuracies.append(acc)
        
        return torch.stack(seq_accuracies)

    def shifted_accuracy(self, a, b, mask):
        batch_size = a.size(0)
        best_accuracy = torch.zeros(batch_size, device=a.device)

        for shift in range(-self.max_shift, self.max_shift + 1):
            if shift < 0:
                shifted_a = a[:, :shift]
                shifted_b = b[:, -shift:]
                shifted_mask = mask[:, :shift] & mask[:, -shift:]
            elif shift > 0:
                shifted_a = a[:, shift:]
                shifted_b = b[:, :-shift]
                shifted_mask = mask[:, shift:] & mask[:, :-shift]
            else:
                shifted_a = a
                shifted_b = b
                shifted_mask = mask

            acc = self.elementwise_accuracy(shifted_a, shifted_b, shifted_mask)
            best_accuracy = torch.max(best_accuracy, acc)

        return best_accuracy

    def levenshtein_distance(self, a, b, mask_a, mask_b):
        batch_distances = []
        a_cpu = a.cpu().numpy()
        b_cpu = b.cpu().numpy()
        mask_a_cpu = mask_a.cpu().numpy()
        mask_b_cpu = mask_b.cpu().numpy()

        for seq1, seq2, m1, m2 in zip(a_cpu, b_cpu, mask_a_cpu, mask_b_cpu):
            seq1_valid = seq1[m1.astype(bool)]
            seq2_valid = seq2[m2.astype(bool)]
            batch_distances.append(self._levenshtein(seq1_valid, seq2_valid))
        return torch.tensor(batch_distances, dtype=torch.float32, device=a.device)

    def _levenshtein(self, seq1, seq2):
        m, n = len(seq1), len(seq2)
        dp = np.zeros((m + 1, n + 1), dtype=int)

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[m][n]

    def global_elementwise_accuracy(self, a, b, mask):
        correct = (a == b) & mask
        total_valid = mask.sum()
        if total_valid == 0:
            return torch.tensor(0.0, device=a.device)
        return correct.sum().float() / total_valid.float()

    def stepwise_accuracy(self, preds, targets, vocab_size=None, token_names=None, figsize=(12, 6)):
        mask = targets != self.pad_token
        correct = (preds == targets) & mask

        token_correct = {}
        token_total = {}

        unique_tokens = torch.unique(targets[mask])
        if vocab_size is not None:
            all_tokens = torch.arange(vocab_size, device=targets.device)
        else:
            all_tokens = unique_tokens

        for token_id in all_tokens:
            token_mask = (targets == token_id) & mask
            correct_preds = correct[token_mask].sum()
            total_preds = token_mask.sum()

            if total_preds > 0:
                acc = correct_preds.float() / total_preds.float()
            else:
                acc = torch.tensor(float('nan'), device=targets.device)

            token_correct[int(token_id)] = acc.item()
        
        return token_correct


    def plot_stepwise_acc(self, preds, targets, figsize=(12, 6), img_path=None):

        token_correct = self.stepwise_accuracy(preds, targets)

        token_ids = list(token_correct.keys())
        accuracies = [token_correct[k] for k in token_ids]

        plt.figure(figsize=figsize)
        sns.heatmap(
            np.array(accuracies).reshape(1, -1),
            annot=True, fmt=".2f", cmap="viridis",
            xticklabels=[self.token_names.get(k, str(k)) if self.token_names else str(k) for k in token_ids],
            yticklabels=["Accuracy"],
            cbar=True
        )
        plt.title("Token-wise Accuracy")
        plt.xlabel("Tokens")
        plt.yticks(rotation=0)

        if img_path:
            img_path = Path(img_path) / "stepwise_acc.png"
            plt.savefig(img_path.as_posix(), bbox_inches='tight', dpi=1200)

        plt.show()


    def topk_most_common_errors(self, preds, targets, k=10):

        mask = (targets != self.pad_token) & (targets != self.stop_token)

        errors = preds[mask] != targets[mask]
        preds_errors = preds[mask][errors]
        targets_errors = targets[mask][errors]

        error_pairs = list(zip(targets_errors.cpu().tolist(), preds_errors.cpu().tolist()))

        from collections import Counter
        counter = Counter(error_pairs)

        most_common = counter.most_common(k)

        return most_common


    def compare(self, a, b):
        """
        compare two sequences for all implemented metrics.
        
        Args:
            a, b: Tensoren der Form (batch_size, seq_len)
        
        Returns:
            dict mit lokalen und globalen Metriken
        """
        assert a.shape == b.shape # ensure both sequences have the same shape

        mask_a = self._create_mask(a)
        mask_b = self._create_mask(b)
        mask = mask_a & mask_b  # compare only non trivial tokens


        local_exact_match = ((a == b) | (~mask)).all(dim=1)
        local_elementwise_accuracy = self.elementwise_accuracy(a, b, mask)
        local_shifted_accuracy = self.shifted_accuracy(a, b, mask)
        local_levenshtein = self.levenshtein_distance(a, b, mask_a, mask_b)

        metrics = {
            "global_elementwise_accuracy": self.global_elementwise_accuracy(a, b, mask),
            "global_exact_match_rate": local_exact_match.float().mean(),
            "exact_match": local_exact_match,
            "elementwise_accuracy": local_elementwise_accuracy,
            "shifted_accuracy": local_shifted_accuracy,
            "levenshtein_distance": local_levenshtein
        }

        return metrics
