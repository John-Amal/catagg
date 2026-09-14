# Releasing

`catagg` publishes to PyPI from GitHub Actions using
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/). No API token
is created, stored or rotated.

## One-off setup

Done once, before the project exists on PyPI.

### 1. PyPI account

Create an account on [pypi.org](https://pypi.org) and enable two-factor
authentication. Uploads are refused without it.

### 2. Register a pending publisher

PyPI lets you configure a publisher for a project that does not exist yet; the
first successful workflow run creates the project and claims the name.

Go to **Account settings → Publishing → Add a new pending publisher** and enter:

| Field | Value |
|---|---|
| PyPI project name | `catagg` |
| Owner | `John-Amal` |
| Repository name | `catagg` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

These must match exactly. A mismatch in any field makes the OIDC token invalid
and the upload fails with a permissions error rather than a helpful one.

### 3. Create the GitHub environment

In the repository: **Settings → Environments → New environment**, named `pypi`.

Optionally add yourself as a required reviewer. That turns every publish into a
manual approval — worth it, given that an upload can never be undone.

## Claiming the name

Run the workflow manually once, from **Actions → Release → Run workflow** on
`main`, while the version in `src/catagg/__init__.py` is still `0.1.0.dev0`.

This uploads a pre-release, which:

- claims the name `catagg` permanently;
- leaves `0.1.0` free for the real release;
- is invisible to `pip install catagg`, since pip ignores pre-releases unless
  `--pre` is passed.

Confirm at <https://pypi.org/project/catagg/>.

## Normal release

1. Set the version in `src/catagg/__init__.py`. It is the single source; the
   build reads it, and a test asserts it matches the distribution metadata.
2. Move the `[Unreleased]` entries in `CHANGELOG.md` under the new version
   heading with its date.
3. Merge to `main` and wait for CI to pass.
4. Tag and push: `git tag v0.1.0 && git push origin v0.1.0`.
5. Publish the GitHub Release for that tag. The workflow refuses to run if the
   tag and the package version disagree.
6. Mint the Zenodo DOI and add it to the README's citation section.

## Irreversibility

A version, once uploaded, can never be uploaded again — deleting a release does
not free its filename. Check the version number before publishing, not after.
