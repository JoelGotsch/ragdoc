---
name: test-writing-guidelines
description: Write tests in a Python file following the project's conventions. Use this when writing, reviewing, or refactoring tests in this project. Follow TDD: when adding a new feature (or changing code), first write a (usually failing) test that captures the expected behaviour, then implement/ update the feature to make the test pass.
---

# Testing Guidelines

Use this file when writing, reviewing, or refactoring tests in this project.
It captures the conventions agreed on during the 2026-04 test suite review and
should be consulted before adding new test files or classes, and when deciding
whether a test is worth keeping.

---

## Structure

- **No class grouping.** Only use `class Test*` when the class has genuine shared state via
  `setup_method`/`teardown_method`. Namespace grouping belongs in section comments, not classes.
  Flat functions with descriptive names are easier to run individually and read at a glance.

- **Unit and integration are separate files.** A unit test is pure logic — no I/O, no real
  documents, deterministic. An integration test operates on real fixtures (parsed files, API
  responses). Never mix them in one file. Use `_integration.py` suffix.

- **Integration tests use 1–3 rich fixtures, tested thoroughly.** Define a small number of
  realistic fixtures (conftest or module-level) and write many focused assertions against them.
  This makes it easy to add a new assertion without setting up new data, and gives a reader
  confidence that difficult real-world cases are covered.

## Async

- **Always use `@pytest.mark.anyio`.** Never call `asyncio.run()` inside a test function.
  Mark the function `async def` and let pytest-anyio manage the event loop.

- **Module-level `asyncio.run()` is allowed only to feed `@pytest.mark.parametrize`.**
  Parametrize arguments must be known at collection time, before any event loop starts.
  This is the one accepted exception. Mark it with a comment explaining why.

## Markers

- **`@pytest.mark.xfail` is a first-class tool.** When a feature is in-progress or a known
  gap exists in the TODO, write a failing test and mark it `xfail(strict=False, reason="...")`.
  This turns the todo list into executable spec and lets you track progress via `pytest -v`.

- **`@pytest.mark.parametrize` over repeated test bodies.** When 3+ test functions differ only
  by input/expected values, collapse them into one parametrized test. Keeps the suite lean.

## Diagnostic scripts vs test internals

- **Tests assert behaviour; scripts explain failures.** When an integration test fails, the
  assertion message should tell the developer which diagnostic script to run — not reproduce the
  script's logic inline.

- **Do not peek into internals to produce test output.** Accessing private scoring functions,
  candidate lists, or text-node indexes inside a test couples it to implementation details and
  makes refactoring harder. If you need that information for debugging, put it in a script under
  `scripts/`.

- **Failure messages should point to the script:**
  ```python
  pytest.fail(
      f"[{case}] footnote not resolved. "
      "Run: uv run python scripts/diagnose_footnotes.py --doc <name>"
  )
  ```

## What not to test

- **Do not test Pydantic field assignment.** If a test only constructs a model and asserts
  that `model.field == value_passed_to_constructor`, it is testing Pydantic, not your code.
  Remove it.

- **Do not test language/framework guarantees.** Abstract methods that raise `NotImplementedError`,
  default return values of builtins, etc. do not need coverage.

- **Do not test generic plumbing (ABCs, pipelines, chaining).** If a class is just glue —
  e.g. "subclass must implement X", "pipeline runs processors in order", "`.add()` returns self" —
  skip the dedicated unit tests. Instead, ensure there is at least one integration test that
  exercises a challenging, realistic input end-to-end through the pipeline with strict assertions.
  That test proves the plumbing works while also catching real bugs. Plumbing-only tests add
  maintenance cost without providing additional safety.

## After writing or changing tests

- **Always run the affected tests.** After adding or modifying tests, run them before finishing:
  ```bash
  uv run pytest tests/test_foo.py          # single file
  uv run pytest tests/ -k "test_name"      # by name
  ```
  Never leave tests unrun — a test that hasn't been executed is unverified.

## Counts and size

- High test count is a smell if it comes from redundancy, not coverage breadth.
  Before adding a test, ask: does this assert a distinct behaviour that could break independently?

- A test file over ~800 lines is a signal to review whether it is doing too much.
