---
description: Scaffold a new uv workspace project with hatchling builds and 3 member packages
---

# Create New UV Workspace

You are tasked with creating a new uv workspace project from scratch. The workspace will have 3 member packages using hatchling as the build backend and src-layout conventions.

## Step 1: Gather Inputs

Ask the user for the following information. Use sensible defaults where noted.

1. **Target directory path** (required) — the full path where the workspace will be created
2. **Workspace name** (default: directory name) — used in root `pyproject.toml`
3. **Three package names** (default: `core`, `models`, `app`) — the names for the 3 workspace member packages. These should be short, kebab-case names (e.g. `data-models`, `voice-agent`).
4. **Python version** (default: `3.12`)
5. **Initialize git?** (default: yes)

Present the defaults and ask the user to confirm or customize. Wait for their response before proceeding.

## Step 2: Create Directory Structure

Create the following structure. Replace placeholders with user-provided values.

**Convention notes:**
- Directory names use kebab-case (e.g. `data-models/`)
- Python package names use underscores (e.g. `data_models/`)
- The package name in `pyproject.toml` uses kebab-case (e.g. `name = "data-models"`)

```
<target>/
├── pyproject.toml
├── .python-version
├── .gitignore
├── README.md
└── packages/
    ├── <pkg-1>/
    │   ├── pyproject.toml
    │   └── src/<pkg_1>/
    │       └── __init__.py
    ├── <pkg-2>/
    │   ├── pyproject.toml
    │   └── src/<pkg_2>/
    │       └── __init__.py
    └── <pkg-3>/
        ├── pyproject.toml
        └── src/<pkg_3>/
            └── __init__.py
```

### 2a: Root `pyproject.toml`

The root is a workspace coordinator — it is NOT installable (no `[build-system]`).

```toml
[project]
name = "<workspace-name>"
version = "0.1.0"
description = ""
requires-python = ">=<python-version>"
dependencies = []

# No [build-system] — root is not installable, just a workspace coordinator

[tool.uv.workspace]
members = ["packages/*"]

[dependency-groups]
dev = [
    "ruff",
]

[tool.ruff]
line-length = 88
indent-width = 4
target-version = "py<python-version-no-dot>"

[tool.ruff.lint]
select = ["E4", "E7", "E9", "F"]
ignore = []
fixable = ["ALL"]
unfixable = []

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"
```

### 2b: Member `pyproject.toml` (repeat for each package)

Each member uses hatchling with src-layout.

```toml
[project]
name = "<pkg-name>"
version = "0.1.0"
description = ""
requires-python = ">=<python-version>"
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/<pkg_name_underscored>"]
```

**Important:** If a package depends on a sibling workspace member, add both:
- The sibling name to `dependencies` (e.g. `"core"`)
- A `[tool.uv.sources]` entry: `core = { workspace = true }`

For the initial scaffold, leave all member `dependencies` empty. The user will add dependencies later with `uv add`.

### 2c: `.python-version`

```
<python-version>
```

### 2d: `.gitignore`

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/

# Virtual environment
.venv/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# uv
uv.lock
```

Wait — **do NOT gitignore `uv.lock`**. The lockfile should be committed for reproducible installs. Remove the `uv.lock` line from `.gitignore`.

### 2e: `__init__.py` files

Each package gets a minimal `__init__.py`:

```python
"""<pkg-name> package."""
```

### 2f: `README.md`

```markdown
# <workspace-name>

UV workspace with 3 packages:

- `<pkg-1>` — TODO: describe
- `<pkg-2>` — TODO: describe
- `<pkg-3>` — TODO: describe

## Setup

```bash
uv sync --all-packages
```

## Development

```bash
# Run a specific package
uv run --package <pkg-name> python -m <pkg_name>

# Add a dependency to a package
cd packages/<pkg-name> && uv add <dependency>

# Lint and format
uv run ruff check .
uv run ruff format .
```
```

## Step 3: Initialize the Workspace

After creating all files:

1. **Run `uv sync`** from the workspace root to create `uv.lock` and `.venv/`
2. **Verify** by running `uv run --package <first-pkg> python -c "import <first_pkg>; print('OK')"`
3. **If git init requested**, run:
   ```bash
   git init
   git add .
   git commit -m "Initial workspace scaffold"
   ```

## Step 4: Report Results

Tell the user:
- What was created (list the directory tree)
- How to add dependencies: `cd packages/<name> && uv add <package>`
- How to add workspace dependencies between members (add to `dependencies` + `[tool.uv.sources]`)
- How to run code: `uv run --package <name> python -m <module>`

## Important Rules

- **Use the Write tool** to create all files — do NOT use `echo` or heredocs in bash
- **Create directories with `mkdir -p`** before writing files into them
- **Never add `[build-system]` to the root** `pyproject.toml` — it's a workspace coordinator only
- **Always use hatchling** as the build backend for member packages
- **Always use src-layout** (`src/<package_name>/`) for member packages
- **Package names in pyproject.toml** use kebab-case; Python module directories use underscores
- **Do not add inter-package dependencies** in the initial scaffold — let the user add them later
