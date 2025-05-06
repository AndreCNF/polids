# GitHub Copilot Custom Instructions

## Python Version and Typing
- Generate Python code compatible with Python 3.13.
- Use modern Python type hints (e.g., `list[str]`, `dict[str, int]`) directly, avoiding imports from the `typing` module (like `List`, `Dict`) whenever possible (i.e., for features available in Python 3.9+).

## Code Style and Documentation
- Always add Google-style docstrings to Python functions and classes.
- Write descriptive and intuitive variable names (e.g., `detected_language_code` instead of `lang` or `dlc`).
- Ensure code aligns with common `ruff` linting rules and conventions.

## Testing

- Use the `pytest` framework for unit testing.
- Follow these best practices to design effective and maintainable tests:
  1. **Keep Tests Focused and Simple**:
     - Test one behavior per test function to simplify debugging.
  2. **Use Descriptive Naming**:
     - Name test functions clearly to reflect their purpose, e.g., `test_calculate_discount_with_valid_input`.
  3. **Structure Tests Logically**:
     - Organize tests in a `tests/` directory, mirroring the structure of your application code.
     - Use filenames like `test_module_name.py` for clarity.
  4. **Isolate Tests**:
     - Ensure each test is independent and does not rely on the state left by other tests.

- Leverage Pytest features to enhance test quality:
  1. **Fixtures for Reusable Setup**:
     - Use `@pytest.fixture` for reusable setup code, and place common fixtures in a `conftest.py` file.
  2. **Parametrize to Avoid Duplication**:
     - Use `@pytest.mark.parametrize` to run a test function with multiple sets of inputs.

- Avoid common pitfalls:
  1. **Overusing Fixtures**:
     - Keep fixtures simple and use them judiciously to maintain test readability.
  2. **Testing Implementation Details**:
     - Focus on testing behavior and outcomes, not internal implementation details.
  3. **Neglecting Edge Cases**:
     - Ensure tests cover edge cases and invalid inputs to verify code robustness.

## Logging and Comments
- Use the `loguru` library for logging. Employ its various log levels appropriately (e.g., `logger.info()`, `logger.warning()`, `logger.error()`, `logger.success()`).
- Write wordy, intuitive, and frequent log messages and comments to explain the code's behavior and state.
- When logging errors or in assertion messages, clearly display both expected and actual values to aid in debugging.

## Data Visualization
- For creating data visualizations and plots, prefer using `plotly`, and specifically `plotly.express` when suitable for concise plot generation.

## Class Design
- When designing Python classes, aim to first define an abstract base class (ABC) from which other concrete classes can inherit.

## Data Modeling
- When appropriate for data structures, use Pydantic V2 (not older, deprecated versions).
- Leverage Pydantic's field types and validators for clean, automatically validated data classes.

## Reproducibility
- When implementing or utilizing stochastic processes (e.g., machine learning model training, simulations), ensure that random seeds are explicitly set to promote reproducibility.

## Jupyter Notebook Structure
- When generating or outlining Jupyter Notebooks, suggest a well-defined Markdown structure. For example:
  ```markdown
  # Notebook Title

  ## Setup
  ### Import libraries
  ### Set parameters and constants
  ### Define auxiliary functions

  ## Load Data
  ### [Describe data source A]
  ### [Describe data source B]

  ## Data Preprocessing
  ### [Step 1]
  ### [Step 2]

  ## Analysis / Benchmark Experiments
  ### Method A
  #### [Implementation/Results for Method A]
  ### Method B
  #### [Implementation/Results for Method B]

  ## Conclusion