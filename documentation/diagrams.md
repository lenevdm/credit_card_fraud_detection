# Diagrams
This document contains various diagrams created with mermaid.js

### High level project structure
The main directories and their purpose
```mermaid
%%{
  init: {
    'theme': 'base',
    'themeVariables': {
      'primaryColor': '#abb2beff',
      'primaryTextColor': '#000000',
      'primaryBorderColor': '#abb2beff',
      'lineColor': '#424242ff',
      'secondaryColor': '#006100',
      'tertiaryColor': '#fff',
      'fontFamily': 'Ariel, sans-serif"
    }
  }
}%%
graph TD
    A(credit_card_fraud_detection/)
    A --> B(config/ <br> <i>Configuration files</i>)
    A --> C(data/ <br> <i>Dataset storage</i>)
    A --> D(notebooks/ <br> <i>Jupyter notebooks</i>)
    A --> E(results/ <br> <i>Experiment results</i>)
    A --> F(src/ <br> <i>Source code</i>)
    A --> G(tests/ <br> <i>Unit tests</i>)
```
  
______

### Detailed project structure
```mermaid
%%{
  init: {
    'theme': 'base',
    'themeVariables': {
      'primaryColor': '#abb2beff',
      'primaryTextColor': '#000000',
      'primaryBorderColor': '#abb2beff',
      'lineColor': '#424242ff',
      'secondaryColor': '#006100',
      'tertiaryColor': '#fff',
      'fontFamily': 'Ariel, sans-serif"
    }
  }
}%%
graph TD
    A(credit_card_fraud_detection/) 
    A --> B(config/ <br> <i>Model configuration</i>)
    B --> J(model_config.py <br> experiment_config.py)
    A --> C(data/ <br> <i>Dataset storage</i>)
    C --> I(creditcard.csv)
    A --> D(notebooks/ <br> <i>Jupyter notebooks for analysis</i>)
    A ----> E(src/ <br> <i>Source code</i>)
    E --> E1(data/ <br> <i>Data processing</i>)
    E --> E2(models/ <br> <i>Model implementations</i>)
    E --> E3(utils/ <br> <i>Utility functions</i>)
    E --> E4(experiments/ <br> <i>Experiment scripts</i>)
    A --> F(tests/ <br> <i>Unit tests</i>)
    E1 --> G1(data_preparation.py)
    E2 --> G2(baseline_model.py)
    E3 --> G3(mlflow_utils.py <br> statistical_analysis.py <br> visualisation_utils.py)
    E4 --> G4(base_experiment.py <br> base_runs_final.py <br> class_weight_experiment.py <br> comparative_analysis.py <br> ensemble_experiment.py <br> rus_experiment.py <br> smote_experiment.py <br> smoteenn_experiment.py )
    F --> G(test_balancing_techniques.py <br> test_baseline_model.py <br> test_data_preparation.py <br> test_evaluation_metrics.py)
    A --> K(results/ )

```

### Component diagram


```mermaid


classDiagram
%%{
  init: {
    'theme': 'base',
    'themeVariables': {
      'primaryColor': '#abb2beff',
      'primaryTextColor': '#000000',
      'primaryBorderColor': '#000000',
      'lineColor': '#000000',
      'secondaryColor': '#006100',
      'tertiaryColor': '#fff',
      'edgeWidth': 3,
      'edgeStrokeWidge': 3,
      'fontFamily': 'Ariel, sans-serif"
    }
  }
}%%
    %% Core Configuration Classes
    class ModelConfig {
        +INPUT_DIM
        +HIDDEN_LAYERS
        +DROPOUT_RATE
        +LEARNING_RATE
        +USE_LR_SCHEDULER
        +BATCH_SIZE
        +MAX_EPOCHS
        +EARLY_STOPPING_PATIENCE
        +TEST_SIZE
        +VAL_SIZE
        +RANDOM_SEED
        +METRICS_OF_INTEREST
    }
    
    class ExperimentConfig {
        +BASE_EXPERIMENT_NAME
        +N_RUNS
        +CONFIDENCE_LEVEL
        +METRICS_OF_INTEREST
        +class SMOTE
        +class RandomUndersampling
        +class SMOTEENN
        +class ClassWeight
        +class Ensemble
    }
    
    %% Core Data and Model Classes
    class DataPreparation {
        +prepare_data()
        +load_and_split_data()
        -_print_split_info()
    }
    
    class FraudDetectionModel {
        -_build_model()
        +train()
        +evaluate()
    }
    
    %% Experiment Base Class
    class BaseExperiment {
        <<abstract>>
        +run_experiment()
        +log_experiment_params()
        -_aggregate_metrics()
        +compare_with()
        +print_results()
        +preprocess_data()*
    }
    
    %% Technique-Specific Experiment Classes
    class BaselineExperimentFinal {
        +preprocess_data()
        +log_experiment_params()
        +print_results()
    }
    
    class SMOTEExperiment {
        +preprocess_data()
        +log_experiment_params()
    }
    
    class RandomUndersamplingExperiment {
        +preprocess_data()
        +log_experiment_params()
    }
    
    class SMOTEENNExperiment {
        +preprocess_data()
        +log_experiment_params()
    }
    
    class ClassWeightExperiment {
        +preprocess_data()
        +log_experiment_params()
    }
    
    class EnsembleExperiment {
        +preprocess_data()
        +train_models()
        +optimize_threshold()
        +ensemble_predict()
        +evaluate_ensemble()
    }
    
    %% Utility Classes
    class ExperimentTracker {
        +log_parameters()
        +log_metrics()
        +log_visualization_artifacts()
        +create_keras_callback()
    }
    
    class ComparativeAnalysis {
        +run_multiple_techniques()
        +analyze_technique_comparisons()
        +generate_summary_table()
    }
    
    class StatisticalAnalysis {
        +paired_t_test()
        +cohens_d()
        +compare_techniques()
        +adjust_pvalues()
    }
    
    %% Relationships
    DataPreparation -- ModelConfig : "uses"
    FraudDetectionModel -- ModelConfig : "configured by"
    BaseExperiment -- ExperimentConfig : "configured by"
    BaseExperiment -- DataPreparation : "uses"
    BaseExperiment -- FraudDetectionModel : "uses"
    BaseExperiment -- ExperimentTracker : "reports to"
    
    BaseExperiment <|-- BaselineExperimentFinal : "implements"
    BaseExperiment <|-- SMOTEExperiment : "implements"
    BaseExperiment <|-- RandomUndersamplingExperiment : "implements"
    BaseExperiment <|-- SMOTEENNExperiment : "implements"
    BaseExperiment <|-- ClassWeightExperiment : "implements"
    BaseExperiment <|-- EnsembleExperiment : "implements"
    
    ComparativeAnalysis -- BaseExperiment : "runs and analyzes"
    ComparativeAnalysis -- StatisticalAnalysis : "uses"
    ExperimentTracker -- MLflow : "integrates with"
    
    note for BaseExperiment "Template Method Pattern for experiments"
    note for EnsembleExperiment "Combines predictions from multiple models"
    note for ComparativeAnalysis "Runs statistical comparisons between techniques"

```



```mermaid
classDiagram
    %% Restructuring to be less wide with better grouping
    
    %% Configuration Classes
    class ModelConfig {
        +INPUT_DIM
        +HIDDEN_LAYERS
        +LEARNING_RATE
        +BATCH_SIZE
        +RANDOM_SEED
        +METRICS_OF_INTEREST
    }
    
    class ExperimentConfig {
        +BASE_EXPERIMENT_NAME
        +N_RUNS
        +CONFIDENCE_LEVEL
        +class SMOTE
        +class RandomUndersampling
        +class SMOTEENN
        +class ClassWeight
        +class Ensemble
    }
    
    %% Core Data & Model Class
    class DataPreparation {
        +prepare_data()
        +load_and_split_data()
    }
    
    class FraudDetectionModel {
        -_build_model()
        +train()
        +evaluate()
    }
    
    %% Abstract Base Class
    class BaseExperiment {
        <<abstract>>
        +run_experiment()
        +preprocess_data()*
        +compare_with()
    }
    
    %% Technique-Specific Experiment Classes - Group 1
    class BaselineExperimentFinal {
        +preprocess_data()
    }
    
    class SMOTEExperiment {
        +preprocess_data()
    }
    
    class RandomUndersamplingExperiment {
        +preprocess_data()
    }
    
    %% Technique-Specific Experiment Classes - Group 2
    class SMOTEENNExperiment {
        +preprocess_data()
    }
    
    class ClassWeightExperiment {
        +preprocess_data()
    }
    
    class EnsembleExperiment {
        +preprocess_data()
        +train_models()
        +ensemble_predict()
    }
    
    %% Utility Classes
    class ExperimentTracker {
        +log_parameters()
        +log_metrics()
        +create_keras_callback()
    }
    
    class ComparativeAnalysis {
        +run_multiple_techniques()
        +analyze_technique_comparisons()
    }
    
    class StatisticalAnalysis {
        +paired_t_test()
        +cohens_d()
        +adjust_pvalues()
    }
    
    %% Core Relationships - Config & Base Classes
    ModelConfig <-- FraudDetectionModel : "configured by"
    ModelConfig <-- DataPreparation : "uses"
    ExperimentConfig <-- BaseExperiment : "configured by"
    
    %% Base Experiment Relationships
    BaseExperiment <-- DataPreparation : "uses"
    BaseExperiment <-- FraudDetectionModel : "uses"
    BaseExperiment <-- ExperimentTracker : "logs via"
    
    %% Inheritance Relationships - Group 1
    BaseExperiment <|-- BaselineExperimentFinal : "extends"
    BaseExperiment <|-- SMOTEExperiment : "extends"
    BaseExperiment <|-- RandomUndersamplingExperiment : "extends"
    
    %% Inheritance Relationships - Group 2
    BaseExperiment <|-- SMOTEENNExperiment : "extends"
    BaseExperiment <|-- ClassWeightExperiment : "extends"
    BaseExperiment <|-- EnsembleExperiment : "extends"
    
    %% Analysis Relationships
    ComparativeAnalysis --> BaseExperiment : "runs"
    ComparativeAnalysis --> StatisticalAnalysis : "uses"
    
    %% Add notes for clarity
    note for BaseExperiment "Template Method Pattern"
    note for EnsembleExperiment "Combines multiple models"
    note for ComparativeAnalysis "Statistical comparison of techniques"
```
  
  _____________

### Data flow diagram
```mermaid
flowchart TD
    %% Data Sources
    rawData[(Credit Card\nTransaction Data)]
    
    %% Data Preparation
    dataPrep[Data Preparation]
    featureSelection[Feature Selection]
    scaling[Feature Scaling]
    splitData[Train/Val/Test Split]
    
    %% Class Balancing Techniques
    baseline[Baseline\nNo Balancing]
    smote[SMOTE\nOversampling]
    randomUS[Random\nUndersampling]
    smoteENN[SMOTE-ENN\nHybrid]
    classWeight[Class\nWeighting]
    
    %% Model Training
    trainModel[MLP Model Training]
    ensembleTraining[Ensemble Training\nMultiple Models]
    
    %% Evaluation
    evalModel[Model Evaluation]
    metrics[Performance Metrics\nCalculation]
    
    %% Analysis
    statAnalysis[Statistical Analysis]
    comparativeAnalysis[Comparative Analysis]
    visualization[Performance\nVisualization]
    
    %% MLflow Tracking
    mlflow[(MLflow\nExperiment Tracking)]
    
    %% Experiment Results
    results[(Experimental\nResults)]
    
    %% Connect Everything
    rawData --> dataPrep
    dataPrep --> featureSelection
    featureSelection --> scaling
    scaling --> splitData
    
    splitData --> baseline
    splitData --> smote
    splitData --> randomUS
    splitData --> smoteENN
    splitData --> classWeight
    
    %% Model Training flows
    baseline --> trainModel
    smote --> trainModel
    randomUS --> trainModel
    smoteENN --> trainModel
    classWeight --> trainModel
    
    baseline --> ensembleTraining
    smote --> ensembleTraining
    randomUS --> ensembleTraining
    smoteENN --> ensembleTraining
    classWeight --> ensembleTraining
    
    trainModel --> evalModel
    ensembleTraining --> evalModel
    
    evalModel --> metrics
    metrics --> statAnalysis
    metrics --> visualization
    
    statAnalysis --> comparativeAnalysis
    visualization --> comparativeAnalysis
    
    metrics --> mlflow
    visualization --> mlflow
    statAnalysis --> mlflow
    comparativeAnalysis --> mlflow
    
    comparativeAnalysis --> results
    
    %% Styling
    classDef preparation fill:#E1F5FE,stroke:#0288D1,stroke-width:2px
    classDef balancing fill:#E8F5E9,stroke:#388E3C,stroke-width:2px
    classDef training fill:#FFF3E0,stroke:#F57C00,stroke-width:2px
    classDef evaluation fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px
    classDef analysis fill:#FFEBEE,stroke:#D32F2F,stroke-width:2px
    classDef storage fill:#ECEFF1,stroke:#607D8B,stroke-width:2px
    
    class dataPrep,featureSelection,scaling,splitData preparation
    class baseline,smote,randomUS,smoteENN,classWeight balancing
    class trainModel,ensembleTraining training
    class evalModel,metrics evaluation
    class statAnalysis,comparativeAnalysis,visualization analysis
    class rawData,mlflow,results storage
```


### Experiment methodology
```mermaid
flowchart TD
    %% Starting Point
    start([Start Experiment]) --> dataLoad[Load Credit Card Dataset]
    
    %% Data Preparation
    dataLoad --> featureSelection[Feature Selection\n9 Most Important Features]
    featureSelection --> standardize[Standardize Features]
    standardize --> stratifiedSplit[Stratified Train/Val/Test Split\n60/20/20]
    
    %% Experiment Loop
    stratifiedSplit --> experimentLoop{For each technique\nRun 30 times}
    
    %% Technique Selection
    experimentLoop --> baseline[Baseline\nNo Balancing]
    experimentLoop --> smote[SMOTE\nOversampling]
    experimentLoop --> rus[Random Undersampling]
    experimentLoop --> smoteenn[SMOTE-ENN\nHybrid]
    experimentLoop --> classWeight[Class Weight]
    experimentLoop --> ensemble[Ensemble\nCombination]
    
    %% Common Model Architecture
    modelArch[MLP Architecture\n9→64→32→16→1\nDropout=0.3]
    
    %% Connect techniques to model
    baseline --> modelArch
    smote --> modelArch
    rus --> modelArch
    smoteenn --> modelArch
    classWeight --> modelArch
    ensemble --> modelArch
    
    %% Training Process
    modelArch --> training[Train with Adam Optimizer\nBinary Cross-Entropy Loss\nEarly Stopping]
    
    %% Evaluation
    training --> evaluation[Evaluate on Test Set]
    evaluation --> metrics[Calculate Metrics:\nAccuracy, Precision, Recall\nF1, ROC-AUC, AUPRC\nG-Mean, MCC]
    
    %% Statistical Analysis
    metrics --> aggregateStats[Calculate Mean & Confidence\nIntervals Across 30 Runs]
    aggregateStats --> pairwiseComp[Paired T-Tests\nCohen's d Effect Size]
    pairwiseComp --> multipleTesting[Multiple Testing Correction\nBenjamini-Hochberg]
    
    %% Results
    multipleTesting --> results[Generate Comparison Tables\nand Visualizations]
    
    %% MLflow
    metrics --> mlflowLogging[Log All Results in MLflow]
    mlflowLogging --> experimentLoop
    
    %% End
    results --> conclusions[Draw Conclusions\nAbout Balancing Techniques]
    results --> visualization[Create Visualizations\nfor Research Paper]
    
    %% Subgraphs for clarity
    subgraph "Data Preparation"
        dataLoad
        featureSelection
        standardize
        stratifiedSplit
    end
    
    subgraph "Technique Comparison"
        baseline
        smote
        rus
        smoteenn
        classWeight
        ensemble
    end
    
    subgraph "Model Training & Evaluation"
        modelArch
        training
        evaluation
        metrics
    end
    
    subgraph "Statistical Analysis"
        aggregateStats
        pairwiseComp
        multipleTesting
        results
    end
    
    %% Styling
    classDef preparation fill:#E1F5FE,stroke:#0288D1,stroke-width:1px
    classDef techniques fill:#E8F5E9,stroke:#388E3C,stroke-width:1px
    classDef training fill:#FFF3E0,stroke:#F57C00,stroke-width:1px
    classDef evaluation fill:#F3E5F5,stroke:#7B1FA2,stroke-width:1px
    classDef analysis fill:#FFEBEE,stroke:#D32F2F,stroke-width:1px
    
    class dataLoad,featureSelection,standardize,stratifiedSplit preparation
    class baseline,smote,rus,smoteenn,classWeight,ensemble techniques
    class modelArch,training training
    class evaluation,metrics evaluation
    class aggregateStats,pairwiseComp,multipleTesting,results,visualization,conclusions analysis
```