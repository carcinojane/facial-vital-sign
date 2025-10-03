# GitHub Repository Preparation Checklist

This guide helps you prepare this repository for pushing to GitHub while handling large dataset files properly.

## ✅ Pre-Push Checklist

### 1. Verify .gitignore is Working

```bash
# Check what will be committed
git status

# Make sure these are NOT listed:
# - data/PURE/** (dataset files)
# - data/UBFC/** (dataset files)
# - *.csv (result files)
# - *.avi, *.mp4 (video files)
# - results.txt (large reports)
```

### 2. Test .gitignore

```bash
# Add a test file in ignored directory
echo "test" > data/PURE/test.txt

# Check it's ignored
git status
# Should NOT show data/PURE/test.txt

# Clean up
rm data/PURE/test.txt
```

### 3. Files That SHOULD Be Committed

```bash
# Code files
scripts/*.py
*.py (root level scripts)

# Configuration
env.yml
requirements.txt
setup.sh

# Documentation
README.md
DATASETS.md
CLAUDE.md
REPOSITORY_DESCRIPTION.md

# Data directory structure (but not datasets)
data/README.md
data/.gitkeep (if exists)

# Git configuration
.gitignore
```

### 4. Files That Should NOT Be Committed

```bash
# Datasets (>20 GB)
data/PURE/**
data/UBFC/**
*.avi
*.mp4
*.zip

# Results (can be large)
results.txt
*.csv
*_results_*.csv
*_evaluation_report.txt

# Plots
*.png
*.jpg
```

## 🚀 Initial Repository Setup

### Step 1: Initialize Git (if not already done)

```bash
cd rppg-vscode-starter
git init
```

### Step 2: Verify .gitignore

```bash
cat .gitignore

# Should contain:
# - data/PURE/**
# - data/UBFC/**
# - *.csv
# - results.txt
# etc.
```

### Step 3: Stage Files Carefully

```bash
# Stage specific directories (safer than git add .)
git add scripts/
git add *.py
git add *.md
git add *.yml
git add *.txt  # Be careful with this - exclude if results.txt exists
git add .gitignore

# Alternatively, use git add . but verify with git status first
git status  # Review everything that will be added
```

### Step 4: Verify No Large Files

```bash
# Check size of staged files
git diff --cached --stat

# OR use git-sizer if available
git-sizer --verbose

# Files should be < 100 MB each
# Total should be < 500 MB
```

### Step 5: Create First Commit

```bash
git commit -m "Initial commit: rPPG vital signs monitoring system

- Complete facial vital signs monitoring implementation
- Support for PURE and UBFC-rPPG datasets
- Incremental improvement evaluation framework
- Multiple UI options (basic, advanced, dashboard)
- Comprehensive documentation"
```

## 🔗 Pushing to GitHub

### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `facial-vital-sign` (or your choice)
3. Description: "Remote photoplethysmography (rPPG) system for contactless vital signs monitoring"
4. **DO NOT** initialize with README (you already have one)
5. Click "Create repository"

### Step 2: Link Remote and Push

```bash
# Add remote
git remote add origin https://github.com/yourusername/facial-vital-sign.git

# Verify remote
git remote -v

# Push to GitHub
git push -u origin master
# OR if using main branch:
git push -u origin main
```

## 🛡️ Safety Checks Before Push

### Check 1: No Large Files

```bash
# List files larger than 10 MB
find . -type f -size +10M | grep -v ".git" | grep -v "data/"

# If any are found and not in .gitignore, add them
```

### Check 2: No Sensitive Data

```bash
# Search for potential secrets
grep -r "password" --exclude-dir=".git"
grep -r "api_key" --exclude-dir=".git"
grep -r "secret" --exclude-dir=".git"
```

### Check 3: Verify .gitignore Coverage

```bash
# Test that datasets are ignored
ls data/PURE/ 2>/dev/null && echo "PURE exists"
git status | grep "data/PURE" && echo "WARNING: PURE not ignored!" || echo "✓ PURE ignored"

ls data/UBFC/ 2>/dev/null && echo "UBFC exists"
git status | grep "data/UBFC" && echo "WARNING: UBFC not ignored!" || echo "✓ UBFC ignored"
```

## 📝 Post-Push Setup

### Add GitHub Repository Details

1. **Add Topics** (on GitHub web):
   - `rppg`
   - `heart-rate`
   - `computer-vision`
   - `opencv`
   - `medical-imaging`
   - `vital-signs`
   - `python`

2. **Update Repository Description**:
   ```
   Remote photoplethysmography (rPPG) system for contactless vital signs monitoring using facial video analysis. Supports multiple datasets and includes incremental improvement framework.
   ```

3. **Add Website** (if applicable):
   - Link to documentation or demo

### Create GitHub Release

After successful push:

```bash
# Tag first version
git tag -a v1.0.0 -m "Initial release: Complete rPPG vital signs monitoring system"
git push origin v1.0.0
```

## 🔄 Future Updates

### Before Each Push

1. **Check for large files**:
   ```bash
   git status
   du -sh .git  # Monitor repo size
   ```

2. **Clean up result files**:
   ```bash
   # Remove generated results (they're in .gitignore)
   rm -f results.txt
   rm -f *_results_*.csv
   rm -f *.png
   ```

3. **Commit specific changes**:
   ```bash
   git add scripts/specific_file.py
   git commit -m "Update: specific change description"
   git push
   ```

## ⚠️ Common Issues

### Issue: "File too large" error

**Solution:**
```bash
# Remove large file from git history
git rm --cached path/to/large/file
git commit --amend
```

### Issue: Dataset accidentally staged

**Solution:**
```bash
# Unstage the dataset
git reset HEAD data/PURE/
git reset HEAD data/UBFC/

# Verify .gitignore includes them
echo "data/PURE/**" >> .gitignore
echo "data/UBFC/**" >> .gitignore
```

### Issue: Results files keep appearing

**Solution:**
```bash
# Add to .gitignore
echo "results.txt" >> .gitignore
echo "*.csv" >> .gitignore

# Remove from git if already tracked
git rm --cached results.txt
git rm --cached *.csv
git commit -m "Remove generated results from tracking"
```

## 📊 Repository Size Guidelines

GitHub recommendations:
- **Repository size**: < 1 GB (ideally < 500 MB)
- **Individual file size**: < 100 MB (hard limit)
- **Recommended file size**: < 50 MB

Our repository without datasets:
- **Estimated size**: ~50-100 MB
- **Code**: ~5 MB
- **Documentation**: ~1 MB
- **Dependencies**: Defined in requirements, not committed

## 🎯 Final Verification

Before pushing, verify:

```bash
# 1. Check total size
du -sh .

# 2. Check git size
du -sh .git

# 3. List all tracked files
git ls-files

# 4. Verify datasets are NOT in the list
git ls-files | grep -i "pure\|ubfc" && echo "WARNING: Datasets found!" || echo "✓ No datasets"

# 5. Count tracked files
git ls-files | wc -l
# Should be < 200 files for this project
```

## ✅ Ready to Push Checklist

- [ ] .gitignore includes all dataset patterns
- [ ] Datasets are NOT in git status
- [ ] Result files are NOT in git status
- [ ] Repository size < 500 MB
- [ ] No files > 100 MB
- [ ] README.md includes dataset download instructions
- [ ] DATASETS.md exists and is complete
- [ ] All code files are included
- [ ] Documentation is up to date

---

**Once all checks pass, you're ready to push to GitHub! 🚀**

```bash
git push -u origin master
```
