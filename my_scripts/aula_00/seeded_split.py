import random

def train_test_split(items, test_fraction=0.2, seed=0):
    if not 0.0 <= test_fraction <= 1.0:
        raise ValueError(f'test_fraction must be between 0.0 and 1.0, got {test_fraction}')
    if len(items) < 2:
        raise ValueError(f'items must have at least 2 elements, got {len(items)}')
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    n_test = min(max(1, int(len(items) * test_fraction)), len(items) - 1)
    test = [items[i] for i in order[:n_test]]
    train = [items[i] for i in order[n_test:]]
    return train, test

items = list(range(10))
print(train_test_split(items, seed=1))
print(train_test_split(items, seed=1) == train_test_split(items, seed=1))
print(train_test_split(items, seed=2)[1])
print(items)
