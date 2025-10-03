# Quick Start: Generate Your Report with Claude

**TL;DR**: Copy the prompt below, paste into Claude.ai, attach your documentation files, get your report!

---

## 🚀 Fastest Way to Generate Report

### **Step 1**: Open Claude.ai
Go to https://claude.ai and start a new conversation

### **Step 2**: Use This Prompt

Copy and paste this into Claude:

```
I'm writing a capstone project report on rPPG (remote photoplethysmography) heart rate monitoring with systematic improvement methodology.

PROJECT SUMMARY:
- Built contactless heart rate monitoring system using facial video
- Baseline: POS algorithm achieved MAE = 4.57 BPM on PURE dataset (24 subjects)
- Systematic approach: Tested 7 improvement methods via ablation study
- Best method: Multi-ROI (forehead + both cheeks) showed 10.4% improvement
- Final result: [Will update with Iteration 2 results]

I have comprehensive documentation of:
✅ Complete iteration log (Iteration 0: Baseline, Iteration 1: Testing, Iteration 2: Implementation)
✅ All evaluation results with metrics
✅ Hypothesis → Implementation → Results → Analysis for each step
✅ Subject-by-subject breakdowns
✅ Code with documentation

I need help writing report sections. Let's start with the ABSTRACT (250-300 words, academic tone, undergraduate capstone level).

Include:
- Problem: rPPG accuracy varies, need systematic improvement
- Approach: Iterative testing of 7 methods from literature
- Key finding: Multi-ROI most effective
- Result: [X]% improvement from baseline
- Conclusion: Systematic methodology validated

Here's my baseline performance data:
- MAE: 4.57 ± 4.44 BPM
- RMSE: 7.99 ± 7.09 BPM
- Correlation: 0.323 ± 0.323
- Within 10 BPM: 89.3%
- 11/24 subjects excellent (MAE < 3 BPM)
- 5/24 subjects poor (MAE > 9 BPM)

Best method from ablation study:
- Multi-ROI: -10.4% MAE improvement
- Mechanism: Average signal across forehead + left cheek + right cheek
- Robustness: Reduces local artifact impact

Can you write the abstract?
```

### **Step 3**: Provide Details as Asked

Claude will ask for more information. Have ready to copy/paste:
- `IMPROVEMENT_LOG.md` (sections as needed)
- `results.txt` (baseline data)
- `iteration2_comparison.txt` (when available)
- Code snippets from `simple_rppg_ui.py`

### **Step 4**: Generate Section by Section

After abstract, continue with:
```
Great! Now let's write the INTRODUCTION (3-4 pages).

Include:
1. Background on rPPG (what it is, applications)
2. Problem: Accuracy challenges in rPPG
3. Motivation: Need for systematic improvement
4. Objectives: Test improvement methods, identify best approach
5. Report structure overview

Here's additional context:
[Paste from IMPROVEMENT_LOG.md - Root Cause Analysis section]
```

Continue for each section: Literature Review, Methodology, Results, Discussion, Conclusion.

---

## 📋 Checklist Before Starting

- [ ] Iteration 2 evaluation complete
- [ ] Have `iteration2_comparison.txt` ready
- [ ] Reviewed `IMPROVEMENT_LOG.md` for accuracy
- [ ] Decided on report length/format requirements
- [ ] Know citation style needed (APA, IEEE, etc.)

---

## 🎯 What to Expect

**Time to generate full report**: 30-60 minutes (section by section)
**Quality**: High-quality first draft, may need minor edits
**Effort**: Mostly copy/paste your existing documentation
**Output**: Markdown format (easy to convert to Word/PDF)

---

## 💡 Pro Tip

**Before asking Claude to generate**, prepare this summary:

```
QUICK FACTS SHEET
=================
Baseline MAE: 4.57 BPM
Multi-ROI MAE: [From Iteration 2]
Improvement: [X]%
Dataset: PURE, 24 subjects
Methods Tested: 7 (Motion, Multi-ROI, Detrend, Adaptive BP, Smooth, Outlier, QA)
Best Method: Multi-ROI (-10.4%)
Worst Method: Motion (0%), Detrend (0%), Outlier (0%)
```

Have this ready to paste when Claude asks for numbers!

---

## 📄 Alternative: Attach Files to Claude

If using Claude.ai web interface (supports file uploads):

1. Click the paperclip icon
2. Upload these files:
   - IMPROVEMENT_LOG.md
   - REPORT_SUMMARY.md
   - results.txt
   - iteration2_comparison.txt
3. Use prompt:

```
I've uploaded my complete rPPG project documentation.

Please write a comprehensive capstone report with:
- Abstract
- Introduction (3-4 pages)
- Literature Review (4-5 pages)
- Methodology (5-6 pages)
- Results (6-8 pages)
- Discussion (4-5 pages)
- Conclusion (2-3 pages)

Academic tone, undergraduate level, ~30-35 pages total.

Focus on the systematic iterative improvement methodology and evidence-based decision making.

Start with the Abstract section.
```

---

## ✅ Final Checklist

After Claude generates each section:

- [ ] Check all numbers match your data
- [ ] Verify technical accuracy
- [ ] Ensure consistency between sections
- [ ] Add proper citations (replace [Author, Year])
- [ ] Insert figures/tables where referenced
- [ ] Proofread for grammar/spelling
- [ ] Format according to requirements

---

## 🆘 If You Get Stuck

**Problem**: Claude's response is too generic
**Solution**: Provide more specific data. Copy/paste exact tables and numbers from your log files.

**Problem**: Technical details are wrong
**Solution**: Correct immediately:
```
The POS algorithm description is slightly off. Here's the exact implementation:
[Paste from IMPROVEMENT_LOG.md - Implementation Details]
Please revise.
```

**Problem**: Need different tone/style
**Solution**: Be explicit:
```
Can you rewrite this in a more:
- Technical/formal tone
- Accessible/simple language
- Detailed/comprehensive style
```

---

## 🎓 Best Practices

1. **Generate section-by-section** (better quality than all-at-once)
2. **Keep Claude focused** (one section per conversation thread)
3. **Provide exact data** (don't make Claude guess numbers)
4. **Iterate on feedback** (ask for revisions if needed)
5. **Save as you go** (copy each section to your document)

---

**Ready?** Just copy the Step 2 prompt and start chatting with Claude! 🚀

**Questions?** See full guide in `CLAUDE_REPORT_PROMPT.md`
