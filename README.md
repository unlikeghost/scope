# SCoPE (Support Compression-based Prediction Engine)

**Author:** Jesus Alan Hernadez Galvan

SCoPE is a Python-based tool and library designed for data classification using dissimilarity metrics (such as NCD and CDM) computed via text or sequence compression algorithms (e.g., gzip, zlib, bz2). **It is a training-free model**: instead of relying on a traditional training phase, SCoPE leverages information theory concepts to predict the class of a query by evaluating how "compressible" it is alongside different support samples.

## 🧠 How does the prediction work?

The prediction process in SCoPE revolves around the **SCoPEDistances** architecture. It starts by generating a **Dissimilarity Matrix** combining support samples and the query using dissimilarity metrics based on multiple compression algorithms (e.g., gzip, zlib, bz2). 

Once the matrix is computed, SCoPEDistances uses distance metrics (such as Euclidean distance) and similarity metrics (such as Cosine similarity) to create an ensemble voting system based on the individual decisions of each compressor and metric combination.

Below is a flowchart summarizing the internal prediction pipeline:

```mermaid
flowchart TD
    %% Styling
    classDef input fill:#f9f9f9,stroke:#333,stroke-width:2px,color:#000;
    classDef process fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000;
    classDef metric fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000;
    classDef decision fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000;
    classDef output fill:#fce4ec,stroke:#c2185b,stroke-width:3px,color:#000;

    %% Nodes
    In([Support Samples & Query]):::input
    
    subgraph Compression["1. Compression Phase"]
        DM[Compute Dissimilarity Matrix]:::process
        Comp[Using Metrics e.g., NCD, CDM <br/> via gzip, zlib, bz2]:::process
    end

    subgraph FeatureExtraction["2. Feature Representation"]
        Centroids[Extract Support Cluster Centroids]:::process
        QueryFeat[Extract Query Features]:::process
    end

    subgraph DistanceCalculation["3. Distance Metrics"]
        Euc[Euclidean Distance]:::metric
        Cos[Cosine Distance]:::metric
        Combined[Combined Distance <br/> Euc &times; Cos]:::metric
    end

    subgraph EnsembleDecision["4. Ensemble & Voting"]
        Classifiers[Base Classifiers Predictions <br/> per Compressor & Metric]:::decision
        Vote[Majority Voting System]:::decision
    end

    Out(((Final Predicted Class))):::output

    %% Edges
    In --> DM
    DM -.-> Comp
    DM --> Centroids
    DM --> QueryFeat
    
    Centroids --> Euc
    QueryFeat --> Euc
    Centroids --> Cos
    QueryFeat --> Cos
    
    Euc --> Combined
    Cos --> Combined
    
    Combined --> Classifiers
    Classifiers --> Vote
    Vote --> Out
```

### Dissimilarity Matrix
For each candidate class, SCoPE combines the query sample with the class support set and evaluates all pairwise relationships using a collection of compressors and distance metrics. The resulting pairwise scores are organized into a structured dissimilarity matrix, where each entry represents the dissimilarity between two samples under a specific compressor-metric combination.

This matrix serves as a compact representation of the relational structure between the query and the support examples, capturing both intra-class consistency and query-to-support similarity patterns.

![Dissimilarity Matrix](diagrams/done/dissimilarity%20matrix%20(done).png)

### Prediction Model
The prediction stage operates on the dissimilarity matrices generated for every candidate class. For each class, SCoPE computes multiple distance functions between the query and supports, producing a collection of distance scores.

The minimum score obtained for each feature across all classes is used as a vote. These votes are then aggregated, and the class receiving the highest number of votes is selected as the final prediction. This voting-based strategy allows SCoPE to leverage diverse compressor-distance combinations while remaining robust to noisy or less informative features.
![Prediction Model](diagrams/done/SCoPE%20distances%20(done).png)


*(Note: Spatial evaluation approaches using Convex Hulls, such as `SCoPEPoligon`, are currently planned for future work).*

## 📊 Experimental Results

SCoPE has been evaluated on various molecular datasets such as ClinTox, BACE, and BBBP. For these evaluations, **70% of the data was used for testing/evaluation**, while the remaining **30% was used for parameter search** (experimenting with various numbers of samples for the support set). The following visualizations demonstrate the model's behavior and performance during experiments on the **ClinTox** dataset:

### Dissimilarity Matrix
Visual representation of the pairwise dissimilarities computed between support samples and a query across different compression methods.

![Dissimilarity Matrix](assets/dissimilarity_matrix.png)

### Normalized Confusion Matrix
Evaluation of the overall predictive performance and class balance on the test data:

![Normalized Confusion Matrix](assets/confusion_matrix_normalized.png)

### AUC-ROC Curve
Performance measurement for the classification model at various threshold settings, illustrating its diagnostic ability:

![AUC-ROC](assets/auc_roc.png)

### Voting Analysis (SCoPEDistances)
Shows how different underlying classifiers (based on different combinations of distance metrics and compressors) contribute and vote towards the final class decision:

![Voting Plot](assets/voting_plot.png)

### Multidimensional Breakdown (Spider Plot)
Representation of the multidimensional dissimilarity space of the query with respect to each evaluated class:

![Spider Plot](assets/spider_plot.png)

### Parameter Search Validation: Learning Curve
![Validation Learning Curve](assets/validation_learning_curve.png)

### Ablation Study: Impact of 'keep_similar'
![Keep Similar Ablation](assets/keep_similar_ablation.png)

### Optimizer Stability Scatter

![Optimizer Stability Scatter](assets/optimizer_stability_scatter.png)

### Synergy Matrix: Compressor vs Metric


![Compressor Metric Synergy Heatmap](assets/compressor_metric_synergy_heatmap.png)

### Final Test Evaluation (Multimetric)

![Final Test Evaluation](assets/final_test_evaluation.png)

### Test Error Rates Evolution


![Test Error Rates](assets/test_error_rates.png)

### Generalization Gap: Search vs Test

![Search vs Test Gap](assets/search_vs_test_gap.png)

## 🛠️ Project Structure

- `src/scope/`: Main source code, defining the compression matrix logic (`compression`), prediction methods, and SCoPE model classes.
- `experiments/`: Scripts for execution, optimization, and performance analysis on experimental datasets (e.g., bace, bbbp, clintox).
- `assets/`: Visual resources and images for documentation.
