# flake8-no-module-forward-call

Flake8 plugin that forbids calling `.forward()` on objects (e.g. `model.forward(inputs)`).

Calling `model.forward(inputs)` bypasses `nn.Module.__call__`, so backward hooks used for gradient synchronization in DDP (Distributed Data Parallel) are not run. Always use `model(inputs)` instead.

## Installation

```
pip install -e .
```

## Usage

Enable in flake8 config (e.g. `setup.cfg` or `.flake8`):

```
[flake8]
extend-select = NMF001
```

Or run:

```
flake8 --extend-select=NMF001 .
```
