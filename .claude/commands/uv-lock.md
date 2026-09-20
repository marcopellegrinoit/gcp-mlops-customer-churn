Generate or update the uv lockfile by running `uv lock`.

## Steps

1. Check that `uv` is available: run `which uv`. If not found, tell the user to install it (`brew install uv`) and stop.
2. Check that a `pyproject.toml` exists in the current working directory. If `$ARGUMENTS` is provided, treat it as the path to a directory containing a `pyproject.toml` and `cd` there first.
3. Run `uv lock` (with `--directory $ARGUMENTS` if a path was supplied).
4. Report the exit code. On success, confirm the lockfile was written. On failure, print the full error output and suggest fixes (e.g. version conflicts, missing packages).
