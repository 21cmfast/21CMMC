# Notes for coding agents

`main` is 21cmMC v2, under development; `v1.x` takes v1 bug fixes only. See
`CONTRIBUTING.rst` for setup and where work goes. The generic inference layer
(parameters, priors, the stage graph) belongs in
[yabf](https://github.com/21cmfast/yabf), not here.

Tests come in two tiers. The fast tier (`pytest`) runs on every push without 21cmFAST
installed. Mark tests that need 21cmFAST or a full simulation `@pytest.mark.nightly`;
they run in the nightly workflow (`pytest -m nightly`).

## How we work

### Documentation: scope and audience

- **Docstrings are timeless and written for users.** Describe what the code does and
  how to use it. No internal jargon, no milestone or task names, no references to
  past issues, pull requests or earlier behaviour.
- **The README is short and to the point.**
- **User-facing documentation never refers to internal decisions** or how we got here.
- **Code comments may explain an internal decision** where that helps the next
  reader, linking the issue if there is one. Never use jargon that only you (the
  agent) know from the current session.

### Tests

- **Prefer physically or statistically meaningful tests** to "functionality" tests that
  only raise coverage. Before writing tests, think about which general physical or
  statistical conditions the code must satisfy (limits, symmetries, conservation,
  known analytic cases, calibration of posteriors and estimators), and test those.
- **Use `pytest.mark.parametrize`** instead of repeating near-identical tests.
- **No tests that exist only for coverage.** A test that a method we just defined
  exists, or that a constructor stores its arguments, tests nothing.

### Code

- **Look for an existing implementation before writing your own**, here and in our
  dependencies. When scoping a change, ask whether refactoring existing code would
  make it simpler.
- **Prefer immutable objects with clear transformations**, built with `attrs`
  (frozen classes; `evolve`/methods that return new objects rather than mutating).
- **Prefer composition over inheritance**, unless inheritance is exactly right for
  the job.
- **Keep backwards compatibility** unless the change is for a new major version. If
  keeping it means jumping through many hoops, stop and reconsider.
- **If following these rules makes the code needlessly complicated or ugly, stop and
  ask a human.**

### Before committing

- **Run the linters through pre-commit** (`pre-commit run --all-files`, or install the
  hooks so they run on every commit) and fix what they report.
- **Don't silence a ruff rule on a line** (`# noqa`) or in the config without a very
  good reason; usually, check with a human first.

## Label every pull request

There is no changelog file to update (`CHANGELOG.rst` ends at v1.0.0): GitHub Releases
are the changelog, written from the merged pull requests. So give every pull request a
clear title, and always add labels when you open it; they sort the release notes. Use only
the labels in `.github/labels.yml`, and add a new one there in the same PR if needed.

- **Exactly one `type:` label**: `type: bug`, `type: feature`, `type: testing`,
  `type: performance`, `type: ci`, or a `type: maint: ...` label (`documentation`,
  `refactoring`, `removal`, `style`, `build`, `dependencies`).
- **`API breaking`** whenever the PR breaks the public API.

`status:`, `priority:` and `effort:` labels are for triaging issues; leave them off PRs.
