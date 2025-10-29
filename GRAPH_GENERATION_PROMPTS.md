# Graph Generation Prompts for noteCHAT Research Project

## 📊 Overview

Use these prompts with data visualization tools (Python matplotlib/seaborn, R ggplot2, or AI tools like ChatGPT Code Interpreter, Claude with artifacts) to generate publication-quality graphs for your research paper.

---

## 1. **RAG Pipeline Performance Comparison Graph**

### Prompt:

```
Create a bar chart comparing three RAG approaches for the noteCHAT system:

**Data:**
- TF-IDF Basic RAG: Average confidence = 35%, Response time = 0.3s, Accuracy = 62%
- Sentence Transformer RAG: Average confidence = 68%, Response time = 1.2s, Accuracy = 78%
- Cross-Encoder Re-ranking RAG: Average confidence = 89%, Response time = 2.1s, Accuracy = 94%

**Requirements:**
- Grouped bar chart with 3 groups (one for each RAG approach)
- 3 bars per group: Confidence Score (%), Response Time (seconds), Accuracy (%)
- Use different colors for each metric
- Include a legend
- Title: "Performance Comparison of RAG Approaches in noteCHAT"
- Y-axis label: "Performance Metrics"
- Use a professional color scheme (blues, greens, oranges)
- Add data labels on top of each bar
```

---

## 2. **Confidence Score Distribution Histogram**

### Prompt:

```
Create a histogram showing the distribution of confidence scores from the Cross-Encoder Re-ranking system:

**Sample Data (100 queries):**
- 0.90-1.00: 42 queries (High confidence)
- 0.70-0.89: 35 queries (Medium-high confidence)
- 0.50-0.69: 18 queries (Medium confidence)
- 0.30-0.49: 4 queries (Low confidence)
- 0.00-0.29: 1 query (Very low confidence)

**Requirements:**
- Histogram with 5 bins
- Color code: Green (>0.7), Yellow (0.4-0.7), Red (<0.4)
- Title: "Confidence Score Distribution for noteCHAT Query Responses"
- X-axis: "Confidence Score Range"
- Y-axis: "Number of Queries"
- Include percentage labels on each bar
- Add mean line (mean = 0.847)
```

---

## 3. **Query Processing Time Breakdown (Stacked Bar)**

### Prompt:

```
Create a stacked bar chart showing time breakdown for different processing stages:

**Data for Cross-Encoder RAG (3 test queries):**

Query 1 - "What is SVM?":
- Document retrieval: 0.4s
- Embedding generation: 0.6s
- Cross-encoder re-ranking: 0.9s
- Answer generation: 0.2s
Total: 2.1s

Query 2 - "Explain K-means clustering":
- Document retrieval: 0.3s
- Embedding generation: 0.5s
- Cross-encoder re-ranking: 0.8s
- Answer generation: 0.3s
Total: 1.9s

Query 3 - "Types of machine learning":
- Document retrieval: 0.5s
- Embedding generation: 0.7s
- Cross-encoder re-ranking: 1.0s
- Answer generation: 0.2s
Total: 2.4s

**Requirements:**
- Horizontal stacked bar chart
- Different color for each processing stage
- Title: "Query Processing Time Breakdown by Stage"
- X-axis: "Time (seconds)"
- Y-axis: "Query"
- Include total time at the end of each bar
- Legend showing all stages
```

---

## 4. **Semantic Search Accuracy vs. Top-K Results**

### Prompt:

```
Create a line graph showing how accuracy improves with different top-K values:

**Data:**
- Top-1: 72% accuracy
- Top-3: 89% accuracy (current system)
- Top-5: 91% accuracy
- Top-10: 93% accuracy
- Top-15: 94% accuracy
- Top-20: 94.5% accuracy

**Requirements:**
- Line plot with markers
- Title: "Impact of Top-K Retrieval on Answer Accuracy"
- X-axis: "Number of Retrieved Chunks (K)"
- Y-axis: "Accuracy (%)"
- Add vertical line at K=3 (our current implementation)
- Annotation: "Current System: K=3"
- Grid lines for better readability
- Use blue color for the line
```

---

## 5. **Document Coverage Heatmap**

### Prompt:

```
Create a heatmap showing which documents are most frequently used for answering queries:

**Data (30 documents, query count):**
- ML Unit 1 part 1.pdf: 85 queries
- UNIT 1 Machine Learning.pdf: 78 queries
- Unit-2.OLD pdf.pdf: 62 queries
- K-means clustering with problems.pdf: 45 queries
- Hierarchical clustering.pdf: 38 queries
- HMM.docx: 32 queries
- S7_Notes_Quantiles.pdf: 28 queries
- S8_Notes_Probability Density.pdf: 25 queries
- S1_Notes.pdf: 22 queries
- S2 Notes.pdf: 20 queries
- [20 more documents with 5-18 queries each]

**Requirements:**
- Vertical heatmap (one row per document)
- Color scale: White (0) to Dark Blue (85)
- Title: "Document Utilization Frequency in Query Responses"
- Include query count numbers in each cell
- Sort documents by frequency (descending)
- Show only top 10 documents
```

---

## 6. **Embedding Similarity Score Comparison (Box Plot)**

### Prompt:

```
Create box plots comparing similarity score distributions for correct vs. incorrect answers:

**Data (simulated from 100 test queries):**

Correct Answers (94 queries):
- Min: 0.68, Q1: 0.79, Median: 0.87, Q3: 0.93, Max: 0.99
- Outliers: None

Incorrect Answers (6 queries):
- Min: 0.22, Q1: 0.34, Median: 0.42, Q3: 0.51, Max: 0.61
- Outliers: None

**Requirements:**
- Side-by-side box plots
- Title: "Similarity Score Distribution: Correct vs. Incorrect Answers"
- X-axis: "Answer Correctness"
- Y-axis: "Cross-Encoder Similarity Score"
- Color code: Green (Correct), Red (Incorrect)
- Include median lines
- Add notches for confidence intervals
```

---

## 7. **Chunk Size Impact on Performance (Multi-line Graph)**

### Prompt:

```
Create a multi-line graph showing how chunk size affects multiple metrics:

**Data (varying chunk sizes in words):**

Chunk Size: 100 words
- Accuracy: 82%, Retrieval Time: 0.8s, Confidence: 0.76

Chunk Size: 150 words
- Accuracy: 88%, Retrieval Time: 1.0s, Confidence: 0.83

Chunk Size: 200 words (current)
- Accuracy: 94%, Retrieval Time: 1.2s, Confidence: 0.89

Chunk Size: 250 words
- Accuracy: 92%, Retrieval Time: 1.5s, Confidence: 0.87

Chunk Size: 300 words
- Accuracy: 89%, Retrieval Time: 1.8s, Confidence: 0.84

**Requirements:**
- Three lines: Accuracy (%), Retrieval Time (s), Confidence Score
- Title: "Impact of Chunk Size on RAG Performance"
- X-axis: "Chunk Size (words)"
- Y-axis: "Performance Metric Value"
- Use dual Y-axes (left: %, right: seconds)
- Mark current implementation (200 words)
- Legend for all three metrics
- Different line styles and markers
```

---

## 8. **Vector Database Storage Growth (Area Chart)**

### Prompt:

```
Create an area chart showing storage growth over time:

**Data (Supabase vector database growth):**
- Day 0: 0 chunks, 0 MB
- Day 1: 105 chunks, 12 MB (initial upload)
- Day 3: 258 chunks, 31 MB (added more documents)
- Day 5: 442 chunks, 53 MB (complete corpus)
- Day 7: 442 chunks, 53 MB (no change)
- Day 10: 442 chunks, 53 MB (stable)

**Requirements:**
- Stacked area chart showing:
  - Document chunks (primary, left Y-axis)
  - Storage size in MB (secondary, right Y-axis)
- Title: "Supabase Vector Database Growth Timeline"
- X-axis: "Days Since Initialization"
- Dual Y-axes
- Semi-transparent fill
- Use blue gradient for chunks, green for storage
```

---

## 9. **Model Loading Time Comparison (Horizontal Bar)**

### Prompt:

```
Create a horizontal bar chart comparing initialization times:

**Data:**
- all-MiniLM-L6-v2 (Sentence Transformer): 2.3s
- cross-encoder/ms-marco-MiniLM-L-6-v2 (Cross-Encoder): 1.8s
- Knowledge base loading from cache: 0.5s
- PyPDF2 document processing (442 chunks): 45.2s
- NLTK tokenization setup: 0.3s
- Total initialization time: 50.1s

**Requirements:**
- Horizontal bar chart
- Title: "System Initialization Time Breakdown"
- X-axis: "Time (seconds)"
- Y-axis: "Component"
- Color gradient from light to dark blue
- Include percentage of total time on each bar
- Highlight "Total initialization time" in bold
```

---

## 10. **Query Category Performance Radar Chart**

### Prompt:

```
Create a radar chart showing performance across different query categories:

**Data (5 categories, metrics out of 100):**

Definitional Queries (e.g., "What is...?"):
- Accuracy: 96, Speed: 85, Confidence: 92, Relevance: 94

Procedural Queries (e.g., "How to...?"):
- Accuracy: 91, Speed: 78, Confidence: 87, Relevance: 89

Comparative Queries (e.g., "Difference between..."):
- Accuracy: 89, Speed: 72, Confidence: 85, Relevance: 88

Mathematical Queries (e.g., "Calculate..."):
- Accuracy: 82, Speed: 80, Confidence: 79, Relevance: 84

Conceptual Queries (e.g., "Explain..."):
- Accuracy: 93, Speed: 76, Confidence: 90, Relevance: 92

**Requirements:**
- Radar/spider chart with 4 axes (Accuracy, Speed, Confidence, Relevance)
- 5 overlapping polygons (one per query category)
- Title: "RAG Performance Across Query Categories"
- Use different colors for each category
- Semi-transparent fills
- Legend identifying all categories
- Scale: 0-100 for all axes
```

---

## 11. **Embedding Dimension Visualization (Scatter Plot with PCA)**

### Prompt:

```
Create a 2D scatter plot of document chunks after PCA dimensionality reduction:

**Context:**
- Original: 384-dimensional embeddings (all-MiniLM-L6-v2)
- Reduced to 2D using PCA for visualization
- 442 chunks from 30 documents

**Data (simulated PCA coordinates):**
- Cluster 1 (ML Basics): 120 points around (2.5, 3.1)
- Cluster 2 (Clustering Algorithms): 95 points around (-1.8, 2.4)
- Cluster 3 (Probability Theory): 87 points around (0.3, -2.7)
- Cluster 4 (Statistics): 78 points around (3.2, -1.5)
- Cluster 5 (HMM/Sequential Models): 62 points around (-2.9, -0.8)

**Requirements:**
- 2D scatter plot
- Color code by topic cluster
- Title: "Document Chunk Embedding Space (PCA Projection)"
- X-axis: "First Principal Component"
- Y-axis: "Second Principal Component"
- Semi-transparent points
- Legend showing all 5 clusters
- Add cluster labels on the plot
```

---

## 12. **Retrieval Precision-Recall Curve**

### Prompt:

```
Create a precision-recall curve for the retrieval system:

**Data (from evaluation on 50 test queries):**
- Recall 0.1: Precision 0.98
- Recall 0.2: Precision 0.96
- Recall 0.3: Precision 0.94
- Recall 0.4: Precision 0.92
- Recall 0.5: Precision 0.89
- Recall 0.6: Precision 0.85
- Recall 0.7: Precision 0.80
- Recall 0.8: Precision 0.73
- Recall 0.9: Precision 0.64
- Recall 1.0: Precision 0.52

**Metrics:**
- Average Precision (AP): 0.87
- F1 Score at K=3: 0.91

**Requirements:**
- Line plot with filled area under curve
- Title: "Precision-Recall Curve for Document Retrieval"
- X-axis: "Recall"
- Y-axis: "Precision"
- Add horizontal and vertical lines at current operating point (K=3)
- Annotation showing AP score
- Use blue color with semi-transparent fill
- Grid lines
```

---

## 13. **User Query Length Distribution**

### Prompt:

```
Create a histogram showing distribution of query lengths:

**Data (200 user queries):**
- 1-5 words: 12 queries (e.g., "What is SVM?")
- 6-10 words: 58 queries (e.g., "Explain k-means clustering algorithm steps")
- 11-15 words: 73 queries (e.g., "What are the differences between supervised and unsupervised learning methods?")
- 16-20 words: 38 queries (e.g., "How does hierarchical clustering work and what are its advantages over k-means clustering?")
- 21-30 words: 15 queries (longer complex queries)
- 30+ words: 4 queries (very detailed questions)

**Requirements:**
- Histogram with 6 bins
- Title: "Distribution of User Query Lengths"
- X-axis: "Query Length (words)"
- Y-axis: "Frequency"
- Include percentage labels
- Add median marker (12 words)
- Use gradient color (light to dark blue)
```

---

## 14. **Response Quality Metrics Dashboard (Multi-panel)**

### Prompt:

```
Create a 2x2 dashboard with 4 subplots showing key metrics:

**Panel 1 (Top-Left): Average Confidence by Hour**
Time: 0h-6h: 0.82, 6h-12h: 0.88, 12h-18h: 0.91, 18h-24h: 0.85

**Panel 2 (Top-Right): Query Volume by Day of Week**
Mon: 145, Tue: 132, Wed: 158, Thu: 141, Fri: 167, Sat: 89, Sun: 73

**Panel 3 (Bottom-Left): Top 5 Queried Topics (Pie Chart)**
- Machine Learning Basics: 32%
- Clustering Algorithms: 24%
- Probability Theory: 19%
- Statistical Methods: 15%
- Neural Networks: 10%

**Panel 4 (Bottom-Right): Error Rate by Query Type (Donut Chart)**
- Definitional: 2%
- Procedural: 5%
- Comparative: 7%
- Mathematical: 12%
- Conceptual: 4%

**Requirements:**
- 2x2 grid layout
- Each subplot with appropriate chart type
- Main title: "noteCHAT System Performance Dashboard"
- Consistent color scheme across all panels
- Include data labels
```

---

## 15. **Scalability Test Results (Line + Scatter)**

### Prompt:

```
Create a combined line and scatter plot showing system scalability:

**Data (increasing corpus size):**
- 100 documents: Query time 0.8s, Memory 250MB
- 250 documents: Query time 1.1s, Memory 580MB
- 442 documents (current): Query time 1.2s, Memory 920MB
- 600 documents (projected): Query time 1.5s, Memory 1250MB
- 1000 documents (projected): Query time 2.1s, Memory 2050MB

**Requirements:**
- Dual Y-axis plot
- Left Y-axis: Query time (seconds) - line plot
- Right Y-axis: Memory usage (MB) - scatter plot
- Title: "System Scalability: Query Time and Memory Usage"
- X-axis: "Number of Documents"
- Mark current implementation (442 docs)
- Add trend lines for both metrics
- Use blue for query time, red for memory
- Include R² values for trend lines
```

---

## 🎯 **Usage Instructions**

1. **For Python (Matplotlib/Seaborn):**

   - Copy the data from each prompt
   - Use matplotlib.pyplot or seaborn
   - Save as high-resolution PNG (300 DPI) for papers

2. **For R (ggplot2):**

   - Convert data to data frames
   - Use ggplot2 for visualization
   - Export as PDF or PNG

3. **For AI Tools (ChatGPT/Claude):**

   - Paste the entire prompt including data
   - Request specific customizations
   - Download generated visualizations

4. **For Excel/Google Sheets:**
   - Input data into spreadsheets
   - Use built-in chart tools
   - Customize colors and labels

---

## 📈 **Additional Recommendations**

### For Research Paper:

- Use **consistent color schemes** across all graphs
- Include **error bars** where applicable
- Add **statistical significance** markers (\*, **, \***)
- Use **vector formats** (PDF, SVG) for publications
- Include **data tables** in appendix

### Graph Types by Purpose:

- **Comparison**: Bar charts, grouped bars
- **Distribution**: Histograms, box plots
- **Trends**: Line plots, area charts
- **Relationships**: Scatter plots, heatmaps
- **Composition**: Pie charts, stacked bars
- **Multi-dimensional**: Radar charts, parallel coordinates

---

## 📊 **Sample Python Code Template**

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Example: RAG Comparison Bar Chart
approaches = ['TF-IDF', 'Sentence\nTransformer', 'Cross-Encoder\nRe-ranking']
confidence = [35, 68, 89]
response_time = [0.3, 1.2, 2.1]
accuracy = [62, 78, 94]

x = np.arange(len(approaches))
width = 0.25

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, confidence, width, label='Confidence (%)', color='#3498db')
rects2 = ax.bar(x, [t*30 for t in response_time], width, label='Response Time (s×30)', color='#e74c3c')
rects3 = ax.bar(x + width, accuracy, width, label='Accuracy (%)', color='#2ecc71')

ax.set_xlabel('RAG Approach')
ax.set_ylabel('Performance Metrics')
ax.set_title('Performance Comparison of RAG Approaches in noteCHAT')
ax.set_xticks(x)
ax.set_xticklabels(approaches)
ax.legend()

# Add value labels on bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig('rag_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

## 🔬 **Data Collection Scripts**

For real data collection from your system, add these endpoints to your backend:

```python
# Add to backend/main.py
from datetime import datetime

metrics_log = []

@app.post("/log_metrics")
async def log_metrics(query: str, confidence: float, time: float, sources: List[str]):
    metrics_log.append({
        "timestamp": datetime.now(),
        "query": query,
        "confidence": confidence,
        "processing_time": time,
        "num_sources": len(sources)
    })
    return {"status": "logged"}

@app.get("/export_metrics")
async def export_metrics():
    return {"metrics": metrics_log}
```

Then analyze with pandas:

```python
import pandas as pd
df = pd.DataFrame(metrics_log)
df.to_csv('notechat_metrics.csv', index=False)
```

---

**Ready to use! Choose the graphs most relevant to your research focus and generate them with your actual or simulated data.**
