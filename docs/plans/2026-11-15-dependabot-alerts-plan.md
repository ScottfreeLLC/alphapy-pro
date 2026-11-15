# Dependabot alerts on main (2026-11-15)

113 open alerts, all in `uv.lock`, from ten packages: Pillow, certifi, cryptography,
urllib3, tornado, virtualenv, filelock, black, setuptools and Mako. None is pinned in
`pyproject.toml` below its patched version.

| Package | Locked | Highest patched version an alert needs |
|---|---|---|
| Pillow | 12.2.0 | 13.0.0 |
| certifi | 2025.11.12 | 2026.10.18 |
| cryptography | 46.0.7 | 50.0.0 |
| urllib3 | 2.6.3 | 2.8.0 |
| tornado | 6.5.5 | 6.6.1 |
| virtualenv | 21.2.4 | 21.14.9 |
| filelock | 3.29.0 | 4.2.1 |
| black | 26.3.1 | 26.10.1 |
| setuptools | 82.0.1 | 84.1.0 |
| Mako | 1.3.12 | 1.4.2 |

## Steps

1. On `develop`, `uv lock --upgrade-package` each of the ten; confirm every locked
   version is at or above the patched version. Status: done. Locked: Pillow 13.0.0,
   certifi 2026.10.23, cryptography 50.0.2, urllib3 2.8.0, tornado 6.6.1, virtualenv
   21.15.6, filelock 4.2.10, black 26.10.1, setuptools 84.2.1, Mako 1.4.3; uv also moved
   distlib, packaging and python-discovery.
2. `uv sync` with every extra; full test suite locally; pre-commit. Status: done.
   `105 passed, 33 warnings in 163.03s`; hooks pass.
3. Check the major bumps for breakage: Pillow 13 (matplotlib, chart CNN rendering),
   cryptography 50, filelock 4. Status: done. alphapy imports none of the ten
   directly. matplotlib 3.10.8 writes PNG and JPEG and Pillow 13 reads them back.
   cryptography only comes in through secretstorage (Linux keyring, via twine) and is
   not installed on macOS.
4. Commit, push `develop`, open a pull request, merge to `main` once CI passes.
   Status: pending.
5. Confirm the alerts close on `main`; list any that remain. Status: pending.
