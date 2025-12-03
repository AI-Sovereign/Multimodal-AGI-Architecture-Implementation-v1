## 📝 Pull Request Checklist (Required for Review)

**DO NOT SUBMIT A PR THAT DOES NOT MEET THESE CRITERIA.**

### 1. Code Integrity & Standards
* [ ] I have ensured this PR targets the correct branch (usually `develop`).
* [ ] The code strictly adheres to the established architectural pattern (e.g., using `config_consts.py` for variables).
* [ ] My changes are fully documented inline using docstrings.
* [ ] The change does not introduce any new security vulnerabilities.

### 2. Testing
* [ ] I have written new unit tests (or updated existing ones) in the correct `tests/` sub-folder.
* [ ] All existing tests pass locally (`pytest packages/`).
* [ ] I have tested the change against the validation schema defined in `resources/schema/validation_rules.json`.

### 3. Reviewer Information
* **Type of change:** [ ] Bug fix / [ ] Feature / [ ] Refactor
* **Affected Product Line(s):** [ ] TCS-25 / [ ] TCS-26 / [ ] Quantum Graph / [ ] Shared Utilities
