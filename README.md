# Data Analysis Assistant

A command-line tool for dataset management, exploratory analysis, and visualization.  
Ideal for quick insights, reproducible reporting, and batch dataset handling.

---

## 📦 Features

- 📁 **Dataset Management**: Load, clean, and organize datasets
- 📊 **Data Analysis**: Summarize, describe, and detect patterns
- 📈 **Visualization**: Generate charts and PDF reports
- 🖥️ **CLI Interface**: Interactive terminal commands for easy use

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/fayezshahid/data_analysis_tool.git
cd data_analysis_tool
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python dataset_manager.py
```

---

## 💡 Example Usage

Once the application is running, you can use the `help` command to get started:

```bash
help
```

Example commands:

```bash
load sample_data/sales.csv sales
analyze
visualize
report
```

**Sample dataset available in**: `sample_data/sales.csv`

---

## 📁 Project Structure

```
data-analysis-assistant/
├── README.md
├── requirements.txt
├── dataset_manager.py          # Main entry point
├── core/
│   ├── __init__.py
│   ├── constants.py
│   ├── data_manager.py         # Dataset management operations
│   ├── analyzer.py             # Data exploration and analysis
│   ├── visualizer.py           # Visualization and reporting
│   └── cli_interface.py        # Command-line interface handler
├── storage/
│   ├── datasets/               # Directory for storing datasets
│   └── metadata/               # Directory for metadata files
├── outputs/
│   ├── plots/                  # Generated visualizations
│   └── reports/                # Generated reports
├── sample_data/
│   └── sales.csv               # Sample dataset for testing
└── tests/
    ├── __init__.py
    ├── test_data_manager.py     # Tests for dataset management
    ├── test_analyzer.py         # Tests for data analysis
    └── test_visualizer.py       # Tests for visualization
```

---

## 🧪 Testing

Run all tests:

```bash
pytest tests/
```

Tests are organized per module:
- `tests/tests_data_manager.py` - Dataset management tests
- `tests/tests_analyzer.py` - Data analysis tests  
- `tests/tests_visualizer.py` - Visualization tests

---

## 🔄 Automated Workflows

This project includes automated testing via GitHub Actions:
- **Feature branch testing**: Automatically runs relevant tests when pushing to `feature/*` branches
- **Workflow configuration**: `.github/workflows/run-feature-tests.yml`

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Contact

**Fayez Shahid** - [@fayezshahid](https://github.com/fayezshahid) - fayezshahid167@gmail.com

Project Link: [https://github.com/fayezshahid/data_analysis_tool](https://github.com/fayezshahid/data_analysis_tool)