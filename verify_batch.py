import torch
from transformers.tokenization_utils_base import BatchEncoding

def compare_batches(a, b):
    # If both are BatchEncoding, compare their dicts
    if isinstance(a, BatchEncoding) and isinstance(b, BatchEncoding):
        a = a.data
        b = b.data
    # If both are dicts, compare keys and values recursively
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            print("Keys mismatch")
            return False
        for k in a:
            if not compare_batches(a[k], b[k]):
                print(f"Mismatch for key: {k}")
                return False
        return True
    # If both are tensors, compare values
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.equal(a, b)
    # If both are lists, compare elementwise
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            print("List length mismatch")
            return False
        for i, (x, y) in enumerate(zip(a, b)):
            if not compare_batches(x, y):
                print(f"Mismatch in list at index {i}")
                return False
        return True
    # Otherwise, compare directly
    return a == b

# Usage:
a = torch.load("run1first_batch.pt", weights_only=False)
b = torch.load("run2first_batch.pt", weights_only=False)
print("Comparing first_batch.pt ...", "MATCH" if compare_batches(a, b) else "MISMATCH")