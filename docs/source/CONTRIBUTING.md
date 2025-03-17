(contributing)=

# Contributing to **Cooper**

We encourage contributions to **Cooper**.

Some things that we would like to see in the future, but have not yet had time to implement, are:

TODO:
- More tutorials showing how to use **Cooper** in non-deep learning applications.
- Implementing **Penalty methods** such as the quadratic penalty method; not currently implemented.
- Implementing **Interior-point methods** (not currently implemented).

## How to contribute

Please follow these steps to contribute:

1. If you plan to contribute new features, please first open an issue and discuss the feature with us.

2. Fork the **Cooper** repository by clicking the **Fork** button on the
   [repository page](http://www.github.com/cooper-org/cooper). This creates
   a copy of the **Cooper** repository in your own account.

3. Install Python >= 3.9 locally in order to run tests.

4. `pip` installing your fork from source. This allows you to modify the code
   and immediately test it out:
    ```bash
    git clone https://github.com/YOUR_USERNAME/cooper
    cd cooper
    pip install --editable .  # Without tests.
    pip install --editable '.[test]'  # Matches test environment.
    pip install --editable '.[dev]'  # Matches development environment.
    pip install --editable '.[notebooks]'  # Install dependencies for running notebooks.
    pip install --editable '.[docs]'  # Used to generate the documentation.
    pip install --editable '.[dev, docs]'  # Install all dependencies.
    ```

5. Add the **Cooper** repo as an upstream remote, so you can use it to sync your
   changes.

   ```bash
   git remote add upstream https://www.github.com/cooper-org/cooper
   ```

6. Create a branch where you will develop from:

   ```bash
   git checkout -b name-of-change
   ```

7. Make sure your code passes **Cooper**'s lint and type checks, by running the following from
   the top of the repository:

   ```bash
   pip install pre-commit
   pre-commit run --all
   ```

8. Make sure the tests pass by running the following command from the top of
   the repository:

   ```bash
   pytest tests
   ```

   **Cooper**'s pipeline tests can take a while to run, so if you know the specific test file that covers your changes, you can limit the tests to that; for example:

   ```bash
   pytest tests/multipliers/test_explicit_multipliers.py
   ```

   You can narrow the tests further by using the `pytest -k` flag to match particular test
   names:

   ```bash
   pytest tests/test_cmp.py -k test_cmp_state_dict
   ```

9. Once you are satisfied with your change, create a commit as follows (
   [how to write a commit message](https://chris.beams.io/posts/git-commit/)):

    ```bash
    git add file1.py file2.py ...
    git commit -m "Your commit message"
    ```

   Then sync your code with the main repo:

    ```bash
    git fetch upstream
    git rebase upstream/main
    ```

   Finally, push your commit on your development branch and create a remote
   branch in your fork that you can use to create a pull request from:

    ```bash
    git push --set-upstream origin name-of-change
    ```

10. Create a pull request from the **Cooper** repository and send it for review. The pull request should be aimed at the `dev` branch.

If you have any questions, please feel free to ask in the issue you opened, or reach out via our [Discord server](https://discord.gg/Aq5PjH8m6E).

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting, formatting and import sorting. We ask for type hints for all code committed to **Cooper** and check for compliance with [mypy](https://mypy.readthedocs.io/).
The CI system should check this when you submit your pull requests.
The easiest way to run these checks locally is via the
[pre-commit](https://pre-commit.com/) framework:

```bash
pip install pre-commit
pre-commit run --all-files
```

## Update notebooks

We use [jupytext](https://jupytext.readthedocs.io/) to maintain two synced copies of the notebooks
in `docs/source/notebooks`: one in `ipynb` format, and one in `md` format. The advantage of the former
is that it can be opened and executed directly in Colab; the advantage of the latter is that
it makes it much easier to track diffs within version control.

To create a new notebook which is automatically synced between the two formats, you can run the following command:

```bash
pip install jupytext==1.16.4
jupytext --set-formats ipynb,md:myst path/to/notebook.ipynb

```

Note that `pre-commit` will automatically ensure that the two formats are in sync.

To manually sync them, you can run the following command:

```bash
jupytext --sync docs/source/notebooks/new_tutorial.ipynb
```

The jupytext version should match that specified in
[.pre-commit-config.yaml](https://github.com/cooper-org/cooper/blob/master/.pre-commit-config.yaml).

To check that the markdown and ipynb files are properly synced, you may use the
[pre-commit](https://pre-commit.com/) framework to perform the same check used
by the GitHub CI:

```bash
pip install pre-commit
pre-commit run jupytext --all-files
```

## Update documentation

To rebuild the documentation, install several packages:

```
pip install -e '.[docs]'
```

And then run:

```
sphinx-build -b html docs/source docs/source/build/html -j auto
```

This can take a long time because it executes many of the notebooks in the documentation source;
if you'd prefer to build the docs without executing the notebooks, you can run:

```
sphinx-build -b html -D nb_execution_mode=off docs/source docs/source/build/html -j auto
```

You can then see the generated documentation in `docs/source/build/html/index.html`.

The `-j auto` option controls the parallelism of the build. You can use a number
in place of `auto` to control how many CPU cores to use.


## License

By contributing to **Cooper**, you agree that your contributions will be
licensed under the LICENSE file in the root directory of this source tree.

## Acknowledgements

This CONTRIBUTING.md file is based on the one from [JAX](https://jax.readthedocs.io/en/latest/contributing.html).
