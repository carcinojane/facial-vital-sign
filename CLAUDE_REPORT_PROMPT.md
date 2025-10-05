# How to Generate Your Report with Claude

This guide shows you how to use Claude (claude.ai) to generate a comprehensive academic/technical report from your documented work.

---

## 📋 What You Have Ready

You've already documented everything Claude needs:

1. ✅ **IMPROVEMENT_LOG.md** - Complete iteration history with hypothesis→results→analysis
2. ✅ **REPORT_SUMMARY.md** - Report structure and key achievements
3. ✅ **results.txt** - Baseline performance data
4. ✅ **iteration2_comparison.txt** - Before/after comparison (when complete)
5. ✅ **incremental_results_*.csv** - Detailed ablation study data
6. ✅ **README.md** - System overview
7. ✅ **DATASETS.md** - Dataset information

---

## 🎯 Recommended Approach: Modular Prompts

**Don't ask Claude to write the entire report at once**. Instead, generate it section by section for better quality.

---

## 📝 Prompt Template for Each Section

### **SECTION 1: Abstract**

```
I need help writing an abstract for my rPPG (remote photoplethysmography) heart rate monitoring project.

Context:
- Project: Systematic improvement of rPPG algorithm accuracy
- Baseline: POS algorithm with MAE = 4.57 BPM on PURE dataset (24 subjects)
- Approach: Tested 7 improvement methods using ablation study
- Best method: Multi-ROI (forehead + both cheeks) showed 10.4% improvement
- Final result: [Include Iteration 2 results when available]

Requirements:
- 250-300 words
- Include: Problem, Approach, Key Results, Conclusion
- Academic tone, suitable for undergraduate capstone report
- Emphasize systematic methodology

Here is my detailed improvement log:
[Paste relevant sections from IMPROVEMENT_LOG.md]
```

---

### **SECTION 2: Introduction**

```
Write an introduction chapter for my rPPG heart rate monitoring project report.

Context:
- Topic: Remote photoplethysmography (rPPG) for contactless heart rate monitoring
- Problem: Existing implementations have inconsistent accuracy
- Motivation: Need systematic approach to identify effective improvements
- Research Questions:
  1. What is baseline performance of standard POS algorithm?
  2. Which improvement methods are most effective?
  3. How much can systematic testing improve accuracy?

Structure needed:
1. Background (what is rPPG, why it matters)
2. Problem statement (accuracy challenges)
3. Motivation (why systematic improvement)
4. Research objectives
5. Report organization

Tone: Academic, informative
Length: 3-4 pages (1000-1500 words)

Reference material:
[Paste from IMPROVEMENT_LOG.md - Iteration 0 section and Root Cause Analysis]
```

---

### **SECTION 3: Literature Review**

```
Write a literature review section covering:

1. rPPG Algorithms:
   - POS (Plane-Orthogonal-to-Skin) - Wang et al., 2017
   - CHROM - de Haan et al., 2013
   - Deep learning approaches (PhysNet, etc.)

2. Improvement Techniques (that I tested):
   - Multi-ROI approaches
   - Motion filtering
   - Signal detrending
   - Adaptive filtering
   - Temporal smoothing
   - Outlier rejection

3. Evaluation Datasets:
   - PURE dataset characteristics
   - UBFC-rPPG dataset

For each technique, explain:
- What it is
- Why it should help (hypothesis)
- Prior work using it

I tested these 7 methods:
[Paste from IMPROVEMENT_LOG.md - Iteration 1 individual method descriptions]

Published benchmarks on PURE:
- POS (Original): MAE ~2-3 BPM, r ~0.85-0.90
- CHROM: MAE ~2.5 BPM, r ~0.80-0.85
- PhysNet: MAE ~1.5 BPM, r ~0.90-0.95

Length: 4-5 pages
Tone: Academic, comprehensive
Include: Citations in text (I'll add proper references later)
```

---

### **SECTION 4: Methodology**

```
Write a detailed methodology section for my rPPG improvement study.

Include these subsections:

4.1 Baseline Implementation
- Algorithm: POS (Plane-Orthogonal-to-Skin)
- Face detection: Haar Cascade
- ROI: Forehead only
- Signal processing: Bandpass 0.7-3.0 Hz, FFT-based HR estimation
[Paste from IMPROVEMENT_LOG.md - Iteration 0 Implementation Details]

4.2 Dataset and Evaluation Protocol
- Dataset: PURE (24 subjects)
- Preprocessing: Frame skip = 2
- Metrics: MAE, RMSE, Correlation, Within-10-BPM percentage
- Ground truth: Pulse oximeter
[Paste from DATASETS.md - PURE dataset section]

4.3 Improvement Method Selection
- 7 methods from literature
- Each method description and hypothesis
[Paste from IMPROVEMENT_LOG.md - Iteration 1 Methods Tested]

4.4 Evaluation Methodology
- Phase 1: Baseline establishment (24 subjects)
- Phase 2: Ablation study (10 subjects, each method independently)
- Phase 3: Full validation of best method (24 subjects)

4.5 Implementation Details
- Programming: Python, OpenCV, SciPy
- Multi-ROI implementation specifics
[Paste code snippets from simple_rppg_ui.py with ITERATION comments]

Length: 5-6 pages
Include: Enough detail for reproduction
Tone: Technical, precise
```

---

### **SECTION 5: Results**

```
Write a comprehensive results section with these subsections:

5.1 Baseline Performance (Iteration 0)
Present this data in well-formatted tables:
[Paste from results.txt or IMPROVEMENT_LOG.md - Iteration 0 Results table]

Key findings:
- Overall MAE: 4.57 ± 4.44 BPM
- 11/24 subjects excellent (MAE < 3 BPM)
- 5/24 subjects poor (MAE > 9 BPM)
- Low correlation (r = 0.323) indicates tracking issues

5.2 Ablation Study Results (Iteration 1)
Present comparison table:
[Paste from IMPROVEMENT_LOG.md - Iteration 1 Summary table]

Key findings:
- Multi-ROI: -10.4% improvement (BEST)
- Motion filtering: No effect (dataset-specific)
- Detrending: No effect (redundant with bandpass)
- etc.

5.3 Multi-ROI Implementation Results (Iteration 2)
[When available, paste from iteration2_comparison.txt]
- Before/after comparison
- Subject-by-subject improvements
- Statistical significance

5.4 Comparison with State-of-the-Art
[Paste comparison table from IMPROVEMENT_LOG.md]

Requirements:
- Clear tables for all numerical results
- Interpret each result (not just numbers)
- Highlight key findings in text
- Refer to tables/figures (e.g., "Table 5.1 shows...")
- Length: 6-8 pages
```

---

### **SECTION 6: Discussion**

```
Write a discussion section analyzing the results:

6.1 Why Multi-ROI Works
Explain the mechanism:
[Paste from IMPROVEMENT_LOG.md - "Why Multi-ROI Works" section]

6.2 Why Other Methods Failed
Analyze:
- Motion filtering: No motion in PURE dataset
- Detrending: Already handled by bandpass filter
- Outlier rejection: Too conservative
[Paste from IMPROVEMENT_LOG.md - Methods That Didn't Work]

6.3 Comparison with Published Work
Our results vs baselines:
[Paste comparison table]

Why gap exists:
- Face detection quality
- Preprocessing differences
- Implementation simplifications

6.4 Subject Variability Analysis
- Why some subjects work better than others
- Face orientation, skin tone, lighting factors
[Paste from IMPROVEMENT_LOG.md - Analysis sections]

6.5 Limitations
- Single dataset (PURE)
- Limited to image sequences (not real-time video)
- Face detection method limitations
- Processing speed considerations

6.6 Lessons Learned
[Paste from IMPROVEMENT_LOG.md - Lessons Learned section]

Length: 4-5 pages
Tone: Critical, analytical
Focus: Explaining WHY, not just WHAT
```

---

### **SECTION 7: Conclusion**

```
Write a conclusion chapter summarizing:

1. Problem Recap
   - rPPG accuracy varies, need systematic improvement

2. Approach Summary
   - Systematic iterative testing framework
   - 7 methods evaluated independently
   - Evidence-based selection

3. Key Results
   - Baseline: MAE = 4.57 BPM
   - Best method: Multi-ROI
   - Improvement: [X]% (from Iteration 2)
   - Final MAE: [X] BPM

4. Contributions
   - Systematic comparison of 7 methods on PURE
   - Multi-ROI implementation variant
   - Documented decision-making process
   - Reusable evaluation framework

5. Future Work
   - Test on UBFC dataset (different characteristics)
   - Implement MediaPipe for better face detection
   - Combine Multi-ROI with adaptive bandpass
   - Real-time optimization
   - Deep learning approaches

6. Closing Statement
   - Systematic approach validated
   - Framework useful for future improvements

Length: 2-3 pages
Tone: Summary, forward-looking
```

---

## 🎨 Prompt for Figures/Tables

### **Generate Figure Captions**

```
I need captions for the following figures in my rPPG report:

Figure 1: Multi-ROI visualization (shows forehead and both cheeks highlighted on face)
[Describe what the figure shows]

Figure 2: MAE distribution histogram - Baseline vs Multi-ROI
[I have the data, need professional caption]

Figure 3: Scatter plot - Ground truth vs predicted HR
[What it demonstrates]

Figure 4: Bland-Altman plot for agreement analysis
[Standard medical visualization]

Figure 5: Iteration progression chart (MAE improvement over iterations)
[Shows systematic improvement]

For each figure, write:
- Short title (1 line)
- Detailed caption (2-3 sentences)
- What the reader should observe
Academic style, informative
```

### **Generate Table Captions**

```
Write captions for these tables:

Table 1: Baseline performance metrics (PURE dataset, 24 subjects)
Columns: Metric, Mean, Std Dev
Rows: MAE, RMSE, MAPE, Correlation, Within-5-BPM%, Within-10-BPM%

Table 2: Ablation study results (7 improvement methods)
Columns: Method, MAE Change, % Improvement, Priority
[Data from Iteration 1]

Table 3: Before/After comparison (Baseline vs Multi-ROI)
[Data from Iteration 2]

Table 4: Comparison with published methods
Methods: Our Baseline, Our Multi-ROI, POS Original, CHROM, PhysNet
Metrics: MAE, Correlation

Format: Academic table captions with context
```

---

## 📊 Prompt for Executive Summary (If Needed)

```
Write a 1-page executive summary of my rPPG improvement project suitable for:
- Non-technical stakeholders
- Project overview document
- Presentation abstract

Include:
- What the project does (rPPG heart rate monitoring)
- Why it matters (contactless vital signs)
- What we achieved (systematic improvement framework)
- Key results (X% improvement via Multi-ROI)
- Impact (shows value of principled testing)

Style: Clear, accessible, non-jargon
Length: 1 page (~500 words)

Data:
[Paste key numbers from IMPROVEMENT_LOG.md]
```

---

## 💡 Pro Tips for Best Results

### 1. **Prepare Your Data First**
Before prompting Claude, have these ready to paste:
- Relevant sections from IMPROVEMENT_LOG.md
- Results tables (copy from .txt or .csv files)
- Code snippets with comments
- Benchmark comparisons

### 2. **Be Specific About Format**
Example:
```
Format requirements:
- Use markdown headings (##, ###)
- Include numbered lists for steps
- Use tables for numerical data
- Cite as [Author, Year] (I'll add full references)
- Academic tone, 3rd person
```

### 3. **Iterate on Sections**
If Claude's first attempt isn't perfect:
```
The methodology section is good, but can you:
- Add more detail on the POS algorithm implementation
- Explain the face detection process more clearly
- Include the specific bandpass filter design (4th-order Butterworth, 0.7-3.0 Hz)
```

### 4. **Combine Sections Later**
Generate each section independently, then:
```
I have 7 sections written. Please:
1. Check consistency between sections
2. Ensure smooth transitions
3. Remove any redundancy
4. Verify technical accuracy
5. Suggest overall flow improvements
```

---

## 🎯 Complete Report Prompt (Alternative)

If you want Claude to generate the whole report at once (not recommended for quality, but faster):

```
Generate a complete undergraduate capstone project report on my rPPG heart rate monitoring system.

Project Overview:
- Topic: Systematic improvement of remote photoplethysmography (rPPG) accuracy
- Baseline: POS algorithm, MAE = 4.57 BPM on PURE dataset
- Approach: Tested 7 improvement methods via ablation study
- Best Result: Multi-ROI (forehead + cheeks) gave [X]% improvement
- Methodology: Iterative, evidence-based, systematic

Report Structure:
1. Abstract (300 words)
2. Introduction (3-4 pages)
3. Literature Review (4-5 pages)
4. Methodology (5-6 pages)
5. Results (6-8 pages)
6. Discussion (4-5 pages)
7. Conclusion (2-3 pages)
8. References (I'll add)
9. Appendices (code snippets)

Style: Academic, undergraduate-level, technical but clear
Length: 30-35 pages total
Format: Markdown with proper headings

Here is all my documentation:

=== IMPROVEMENT_LOG.md ===
[Paste entire IMPROVEMENT_LOG.md]

=== Baseline Results ===
[Paste results.txt]

=== Ablation Study Results ===
[Paste incremental_results summary]

=== Iteration 2 Results ===
[Paste iteration2_comparison.txt when available]

=== Dataset Info ===
[Paste from DATASETS.md]

Generate the complete report following academic standards.
```

---

## 📤 After Generation

Once Claude generates sections:

1. **Review for accuracy** - Check all numbers match your data
2. **Add references** - Replace [Author, Year] with proper citations
3. **Add figures** - Insert visualizations where referenced
4. **Check flow** - Read entire document for coherence
5. **Proofread** - Grammar, spelling, formatting

---

## 🔖 Bookmark This

Save these prompts for future use. You can:
- Modify for different sections
- Reuse for other projects
- Share with teammates
- Adapt for different audiences (technical vs non-technical)

---

## 📞 Example Conversation Starter

```
Hi Claude! I'm working on my capstone project report about rPPG (remote photoplethysmography) heart rate monitoring.

I've systematically improved the accuracy through an iterative process:
- Started with baseline POS algorithm (MAE = 4.57 BPM)
- Tested 7 improvement methods using ablation study
- Identified Multi-ROI as best method (10.4% improvement)
- Validated on full 24-subject PURE dataset

I have complete documentation of all iterations with hypothesis, implementation, results, and analysis for each step.

I need help writing the report section by section. Can we start with the abstract? I'll provide you with the key data points.
```

Then paste relevant sections as Claude asks for details.

---

**Last Updated**: 2025-10-02
**Status**: Ready to use once Iteration 2 results are available
**Tip**: Start with smaller sections (abstract, intro) to get comfortable with the process
