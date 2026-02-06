# Kaggle Competitions Repository

A collection of Kaggle competition solutions with reusable data science utilities.

## Repository Structure

```
Kaggle/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── ML_CHECKLIST.md       # Machine Learning project checklist
├── seaborn.md            # Seaborn reference guide
│
├── common/               # Shared utilities across competitions
│   ├── __init__.py
│   ├── utils.py          # General utility functions
│   └── functions/
│       ├── __init__.py
│       └── get_data_insights.py  # EDA helper functions
│
└── house-prices/         # House Prices competition
    ├── README.md
    ├── analysis.ipynb
    ├── model_creation.ipynb
    ├── housing_pipeline.py
    ├── data/
    └── submissions/
```

## Competitions

| Competition | Status | Best Score (Test score)| Description |
|-------------|--------|------------|-------------|
| [House Prices](./house-prices/) | Active | 0.11814 (Leaderboard: 103 of 4380)| Predict house sale prices using regression |

## Common Module

The `common` module provides reusable functions for data science workflows:

## Setup

### Prerequisites

- Python 3.10+

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/kaggle.git
cd kaggle
```

2. Create virtual environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure Kaggle API (for downloading data) - Optional:
```bash
# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or %USERPROFILE%\.kaggle\ (Windows)
kaggle competitions download -c house-prices-advanced-regression-techniques
```

## ML Project Workflow

This repository follows the ML Checklist (see `ML_CHECKLIST.md`) based on "Hands-On Machine Learning" by Aurélien Géron:

1. **Frame the Problem** - Define objectives and metrics
2. **Get the Data** - Download and prepare datasets
3. **Explore the Data** - EDA in Jupyter notebooks
4. **Prepare the Data** - Custom transformers in pipeline files
5. **Model Selection** - Compare multiple algorithms
6. **Fine-tune Models** - Hyperparameter optimization
7. **Present Solution** - Document findings
8. **Launch** - Create submission files

## Best Practices

### Code Organization
- Each competition in its own directory
- Reusable code goes in `common/`
- Custom transformers follow sklearn conventions
- Notebooks for exploration, `.py` files for production code

### Feature Engineering Pipeline
- All transformations as sklearn-compatible classes
- Pipeline handles train/test consistently
- No data leakage between train and test

### Version Control
- Don't commit large data files (use `.gitignore`)
- Commit notebooks with outputs cleared
- Use meaningful commit messages

## Adding a New Competition

1. Create competition directory:
```bash
mkdir new-competition
cd new-competition
```

2. Initialize structure:
```
new-competition/
├── README.md
├── analysis.ipynb      # EDA notebook
├── model.ipynb         # Model training
├── pipeline.py         # Custom transformers
├── data/              
└── submissions/
```

3. Download competition data:
```bash
kaggle competitions download -c competition-name -p data/
```

## License

MIT License - see LICENSE file

## Resources

- [Kaggle Learn](https://www.kaggle.com/learn)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Hands-On ML Book](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
