
# Contributing to **Cooper**


We want to make contributing to **Cooper** as easy and transparent as
possible.


## Building

Using `pip`, you can install the package in development mode by running:

```sh
pip install --editable "[.dev]"
```

## Testing

We test the package using `pytest`, which you can run locally by typing

```sh
pytest tests
```

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.


## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## Coding Style

We use `black` for formatting `isort` for import sorting, `flake8` for
linting. We ask for type hints for all code committed to **Cooper** and check
for compliance with `mypy`. The CI system should check of this when you submit
your pull requests.

## License

By contributing to **Cooper**, you agree that your contributions will be
licensed under the LICENSE file in the root directory of this source tree.
