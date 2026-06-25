
Detailed comparison between the previous run and this new run:

### 1. Key Metrics Comparison

| Metric | Previous Run | New Run | Change |
| :--- | :--- | :--- | :--- |
| **Total Runtime** | 119.28 seconds | **32.85 seconds** | **86.43 seconds faster** (~3.6x speedup) |
| **Total Function Calls** | 1,085,462 | **9,885** | **1,075,577 fewer calls** (~99% reduction) |
| **Subprocess Duration (Cumulative)** | 118.76 seconds | **32.84 seconds** | **85.92 seconds faster** |
| **Average Time per Subprocess Call** | ~39.58 seconds | **~10.95 seconds** | **28.63 seconds faster per call** |

---

### 2. Why Did It Improve?

The improvement comes from two distinct areas:

#### A. Faster Subprocesses (The Main Time Saver)
The program still runs exactly 3 subprocess steps (`run_step` is called 3 times in both profiles). However, the external processes finished much quicker this time:
* **Before:** The 3 processes took a combined **118.76 seconds** (~39.6 seconds each).
* **Now:** The 3 processes took a combined **32.84 seconds** (~10.9 seconds each).
This change alone accounts for almost the entire 86-second reduction in total execution time. 

#### B. Elimination of HTML/Regex Overhead
In the first profile, the script spent time processing HTML data, calling functions like `count_html_tags` and `count_html_fields` over 73,000 times and executing 55,000 regex operations. 
* In this new run, those operations are completely absent. 
* The total function call count dropped from over 1 million down to under 10,000, reducing internal Python overhead to nearly zero.

---

### 3. Current State of the Code

Although the program is running much faster, the primary bottleneck remains the same:
* Out of the 32.85 seconds of total runtime, **32.83 seconds** (99.9%) is spent waiting inside `select.poll` for the subprocesses to complete.
* The local Python execution overhead is now virtually non-existent (taking about 0.017 seconds in total). 

If you need to make the program run even faster, any further optimization would require parallelizing the 3 subprocesses (so they run concurrently rather than sequentially) or optimizing the external tools/commands themselves.