# Customer Revenue Analytics

This project implements a Data Analytics dashboard using Python, Pandas, SQLite, Flask, and Streamlit.

## Project Structure

- `data/`: Contains raw CSV data and the SQLite database.
- `src/etl.py`: ETL script to load CSV data into SQLite.
- `src/api.py`: Flask API to serve data from the database.
- `src/dashboard.py`: Streamlit dashboard to visualize the data.

## Prerequisites

Install the required libraries:

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Initialize the Database
Run the ETL script to load the data:

```bash
python src/etl.py
```

### 2. Start the Flask API
In a terminal, run:

```bash
python src/api.py
```
This will start the API at `http://127.0.0.1:5000`.

### 3. Start the Streamlit Dashboard
Open a **new** terminal and run:

```bash
streamlit run src/dashboard.py
```
This will open the dashboard in your browser.

## Technologies Used
- **Python**: Core language.
- **Pandas**: Data manipulation.
- **SQLite**: Database.
- **Flask**: Backend API.
- **Streamlit**: Frontend Dashboard.
- **Matplotlib/Seaborn**: Visualization.
