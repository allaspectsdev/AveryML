"""Shared test fixtures for AveryML.

Importing the subpackages here ensures registry decorators fire
before any test that uses the registries.
"""

import averyml.evaluation.benchmarks  # noqa: F401
import averyml.synthesis.backends  # noqa: F401
import averyml.synthesis.prompts  # noqa: F401
import averyml.training.backends  # noqa: F401
