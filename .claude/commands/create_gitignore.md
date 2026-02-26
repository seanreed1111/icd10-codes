---
description: Create a robust Python .gitignore at the git repo root
allowed-tools: Bash(git rev-parse:*), Read, Write, Bash(pwd:*)
---

Create a robust `.gitignore` file for Python projects at the git repository root.

## Instructions

Generate and write a `.gitignore` file that covers:

1. **Python artifacts** — bytecode, compiled extensions, cache dirs, eggs, distributions, packaging, virtual environments, installers
2. **Testing & coverage** — pytest cache, coverage reports, tox, mypy, ruff, pytype caches
3. **Dev tools** — VS Code, PyCharm, Jupyter, Spyder, rope
4. **OS files** — macOS `.DS_Store`, Windows `Thumbs.db`, Linux `*~`
5. **Temp & log directories** — `tmp/`, `temp/`, `logs/`, `log/`, and common log file extensions
6. **Secrets** — `.env`, `.env.*`, `*.pem`, `*.key`, `secrets.*`

### Steps

1. Run `git rev-parse --show-toplevel` to find the repository root. If this fails (not a git repo), fall back to `pwd` and warn the user.
2. Check if a `.gitignore` already exists at that root path. If it does, show the user its contents and ask whether to overwrite or append.
3. Write the following content to `<repo-root>/.gitignore` (overwrite or append based on user choice):

```
# ── Python ───────────────────────────────────────────────────────────────────
__pycache__/
*.py[cod]
*$py.class
*.so
*.pyd

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
.python-version
.pdm-python
.pdm.toml
.pdm-build/
__pypackages__/

# uv
.uv/
uv.lock          # remove this line if you want to commit uv.lock

# ── Testing & coverage ────────────────────────────────────────────────────────
.tox/
.nox/
.coverage
.coverage.*
coverage.xml
*.cover
*.py,cover
htmlcov/
.pytest_cache/
.cache/
nosetests.xml
test-results/
junit.xml

# ── Type checkers & linters ───────────────────────────────────────────────────
.mypy_cache/
.dmypy.json
dmypy.json
.pytype/
.pyre/
.ruff_cache/

# ── Jupyter ───────────────────────────────────────────────────────────────────
.ipynb_checkpoints
*.ipynb_checkpoints/
profile_default/
ipython_config.py

# ── Documentation builders ────────────────────────────────────────────────────
docs/_build/
site/
.pdoc/

# ── Editors & IDEs ────────────────────────────────────────────────────────────
# VS Code
.vscode/
*.code-workspace

# JetBrains / PyCharm
.idea/
*.iml
*.iws

# Spyder
.spyderproject.db
.spyproject/

# Rope
.ropeproject

# ── Secrets & credentials ─────────────────────────────────────────────────────
.env
.env.*
!.env.example
*.pem
*.key
*.p12
*.pfx
secrets.*
!secrets.example.*

# ── Temp & scratch ────────────────────────────────────────────────────────────
tmp/
temp/
scratch/
.tmp/
*.tmp
*.bak
*.swp
*.swo

# ── Logs ──────────────────────────────────────────────────────────────────────
logs/
log/
*.log
*.log.*
*.out

# ── OS-generated ──────────────────────────────────────────────────────────────
# macOS
.DS_Store
.AppleDouble
.LSOverride
._*
.Spotlight-V100
.Trashes

# Windows
Thumbs.db
Thumbs.db:encryptable
ehthumbs.db
Desktop.ini
$RECYCLE.BIN/
*.lnk

# Linux
*~
.fuse_hidden*
.directory
.Trash-*
.nfs*
```

4. After writing the file, confirm success and print the absolute path of the file created.
