# mec8211

Repository for all projects (homework, results, final project) for the MEC8200 class. All projets have their own folders for simplicity

---

## General Rules :

* **Always `git pull origin main` before starting work**
* **Only merge to main when tests pass**
* **Commit working code, not broken experiments**
* **Use descriptive branch names:** `justin/source-term-unit-test` not `justin/update-test`
* LLMs are very useful for any git-wise tasks when properly defined in prompt.

---

## Proposed Workflows :

#### 1. Start your work session (do this EVERY time before coding) :
```
git checkout main
git pull origin main          		# Get latest pushed content from teammates
git checkout -b your-name/feature-name  # Create your branch (first time)
# OR
git checkout your-name/feature-name     # Switch to existing branch
git merge main                		# Bring those changes into your work
# Now work on your stuff
```

#### 2. Save your work frequently :
```
git add .				# Stage all changed files
# OR
git add path/to/file 			# Stage specific file

git commit -m "Descriptive message"	# Save staged work with clear message
```

> **Note**
> You can optimize your commit history by splitting work across multiple task oriented commits. For example if you have changes to both unit tests and core code, you could have a commit only for the unit tests and another for the core changes.

#### 3. Share your work when it's ready :
```
python -m pytest tests/			# Make sure tests pass first

git checkout main
git pull origin main 			# Get any new changes
git merge your-name/feature-name        # Merge your work
git push origin main                    # Share changes with team
```

---

## Common Tasks

#### See what you've changed:
```bash
git status                              # What files changed?
git diff                                # What exactly changed?
```

#### Undo uncommitted changes:
```bash
git checkout -- filename.py             # Undo changes to one file
git reset --hard                        # Undo ALL uncommitted changes (careful!)
```

#### See commit history:
```bash
git log --oneline                       # Brief history
git log --graph --oneline --all         # Visual branch history
```

#### Push your branch to GitHub (backup/sharing):
```bash
git push origin yourname/feature-name
```
