# DA Projects

A repository containing a hotel booking cancellation analysis project built with Python and Jupyter notebooks.

## Project Overview

This repository currently includes the `hotel_booking_da` project, which explores hotel booking data, generates automated EDA reports, and prepares dataset preprocessing for modeling.

## Prerequisites

- Python 3.9 or higher
- pip package manager

## Dependencies

The project uses the following Python packages:
- **numpy** - Numerical computing and array operations
- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning algorithms and preprocessing
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization
- **plotly** - Interactive charts in notebooks
- **ydata-profiling** - Automated exploratory data analysis reports

## Installation & Initialization

1. Open a terminal in the repository root:
   ```bash
   cd "c:\Users\madhu\Desktop - Copy\da\DA_Projects"
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. (Optional) Install Jupyter if you want to run notebooks:
   ```bash
   pip install notebook
   ```

## Usage

Use the `hotel_booking_da` folder for the current hotel booking analysis project.

- Run dataset overview and preprocessing helper:
  ```bash
  python hotel_booking_da\preprocess.py
  ```

- Generate the automated EDA report:
  ```bash
  python hotel_booking_da\autoeda.py
  ```

- Open notebooks for interactive analysis:
  ```bash
  jupyter notebook
  ```
  Then open `hotel_booking_da/data_explore.ipynb` or `hotel_booking_da/eda.ipynb`.

## Notes

This project is notebook-first and designed for local analysis. If you expand it later, consider adding model training scripts, dashboards, or API endpoints.

## License

No license is included in this repository at this time.
