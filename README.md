# flake8-no-model-forward-call

[![PyPI version](https://img.shields.io/pypi/v/flake8-no-model-forward-call.svg)](https://pypi.org/project/flake8-no-model-forward-call/)
[![Python versions](https://img.shields.io/pypi/pyversions/flake8-no-model-forward-call.svg)](https://pypi.org/project/flake8-no-model-forward-call/)
[![License](https://img.shields.io/pypi/l/flake8-no-model-forward-call.svg)](https://github.com/mrsndmn/flake8-no-model-forward-call/blob/main/LICENSE)
[![CI](https://github.com/mrsndmn/flake8-no-model-forward-call/actions/workflows/ci.yml/badge.svg)](https://github.com/mrsndmn/flake8-no-model-forward-call/actions/workflows/ci.yml)

A [flake8](https://flake8.pycqa.org/) plugin that forbids calling `.forward()` directly on PyTorch `nn.Module` objects.

## Why?

Calling `model.forward(inputs)` bypasses `nn.Module.__call__`, which means **PyTorch hooks are silently skipped** — including the backward hooks that `DistributedDataParallel` (DDP) relies on for gradient synchronization.

```python
# Bad — DDP backward hooks won't run, gradients won't sync
loss = model.forward(inputs)

# Good — goes through __call__, all hooks fire correctly
loss = model(inputs)
```

This bug is easy to introduce, hard to notice (training may appear to work), and can silently corrupt distributed training runs.

## Installation

```bash
pip install flake8-no-model-forward-call
```

For development / editable install:

```bash
git clone https://github.com/mrsndmn/flake8-no-model-forward-call.git
cd flake8-no-model-forward-call
pip install -e .
```

## Usage

Run flake8 with the check enabled:

```bash
flake8 --extend-select=NMF001 .
```

Or add it to your flake8 config so it runs automatically:

**.flake8** / **setup.cfg**:
```ini
[flake8]
extend-select = NMF001
max-line-length = 128
```

**pyproject.toml**:
```toml
[tool.flake8]
extend-select = ["NMF001"]
max-line-length = 128
```

## Error codes

| Code   | Message |
|--------|---------|
| NMF001 | Do not call `.forward()` directly; use `model(inputs)` instead so DDP backward hooks run |

## Example

```python
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return x * 2

model = MyModel()

model.forward(x)   # NMF001 — flagged
model(x)           # OK
```

Running flake8:

```
example.py:10:1: NMF001 Do not call .forward() directly; use model(inputs) instead so DDP backward hooks run
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b my-feature`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest`
5. Open a pull request

## License

Apache 2.0 — see [LICENSE](LICENSE).
