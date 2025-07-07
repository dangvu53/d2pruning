import torch

def compare_tensors(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            print("Keys mismatch")
            return False
        for k in a:
            if not torch.equal(a[k], b[k]):
                print(f"Tensor mismatch for key: {k}")
                return False
        return True
    elif torch.is_tensor(a) and torch.is_tensor(b):
        return torch.equal(a, b)
    else:
        print("Type mismatch")
        return False

files = [
    "first_grads.pt",
    "first_updated_weights.pt",
    "initial_weights.pt"
]

for fname in files:
    a = torch.load(f"run1{fname}", weights_only=False)
    b = torch.load(f"run2{fname}", weights_only=False)
    print(f"Comparing {fname} ... ", end="")
    if compare_tensors(a, b):
        print("MATCH")
    else:
        print("MISMATCH")