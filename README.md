# Brent Oil Data Analysis

> Brent Oil Data Analysis is a data science project aimed at analyzing historical Brent oil price data. The project cleans the dataset by removing duplicate rows and handling missing values. It performs Exploratory Data Analysis (EDA) and builds statistical models such as ARIMA and GARCH. Additionally, it employs Bayesian analysis and Monte Carlo simulations to analyze price fluctuations. This project is implemented in Python and leverages powerful financial libraries.

## Built With

- Programming Language: Python 3
- Libraries: NumPy, pandas, matplotlib, Seaborn, statsmodels (for ARIMA), ARCH (for GARCH), PyMC (for Bayesian analysis), REACT, Flask, etc
- Tools & Technologies: Jupyter Notebook, Google Colab, Git, GitHub, Gitflow, VS Code

## Demonstration and Website

[Deployment link]()


---
## Project Structure
```bash
brent-data-analysis\
│── .github/workflows/              # GitHub Actions for CI/CD
│   ├── pylint.yml                 # Workflow for checking Python code styles
│── data\                          # Directory for datasets
│   ├── BrentOilPrices.csv         # Raw dataset
│   ├── processed_data.csv         # Processed dataset (after cleaning)
│
│── notebooks\                     # Jupyter notebooks for analysis
    ├── change_point_analyzer.ipynb  # Notebook for change point analysis
│   ├── data_analyzer.ipynb         # Notebook for data analysis
│   ├── statistical_modeling.ipynb  # Notebook for statistical modeling
│   ├── plots/                      # Directory for storing generated plots
│       ├── time_series_plot.png    # Time series plot
│       ├── acf_pacf_plot.png       # ACF and PACF plot
│── backend\                     # Bakend folder for flask endpoints
│   ├── app.py         # the main endpoint app
│── frontend\                     # Jupyter notebooks for analysis
│   ├── node_modules         # dependencies for React
│   ├── public/  # Folder that contain the main homepage
│   ├── src/                      # Directory for components, services, styles
│   ├── .gitignore   # Distinct .gitignore for frontend
│   ├── package.json # Dependencies and libraries
    ├── README.md  # README.md for REACT frontend
│   ├── acf_pacf_plot.png       # ACF and PACF plot


│
│── scripts/                        # Python scripts for different modules
    ├── __init__.py
│   ├── data_analysis.py            # Data cleaning, analysis, and EDA
│   ├── statistical_modeling.py     # Statistical modeling (ARIMA, GARCH, Bayesian)
    ├── lstm.py            # LSTM model building
│   ├── change_point_analysis.py   # Scripts for analyzing change pints using VAR, MARKOV_SWICHING_ARIMA, LSTM
│
│── src/                            # Main automation script
    ├── __init__.py
│   ├── src.py                      # Main pipeline script
│
│── tests              # Folder that contain testing scripts
│   ├── __init__.py
│
│── requirements.txt                 # Dependencies and libraries
│── README.md                        # Project documentation
│── .gitignore                        # Files and directories to ignore in Git
```

---

## Getting Started

You can clone this project and use it freely. Contributions are welcome!

### Cloning the Repository

To get a local copy, run the following command in your terminal:

```sh
git clone https://github.com/amare1990/brent-data-analysis.git
```

Navigate to the main directory:

```sh
cd brent-data-analysis
```

### Setting Up a Virtual Environment

1. Create a virtual environment:
   ```sh
   python -m venv venv-name
   ```
   Replace `venv-name` with your desired environment name.

2. Activate the virtual environment:
   - **On Linux/macOS:**
     ```sh
     source venv-name/bin/activate
     ```
   - **On Windows:**
     ```sh
     venv-name\Scripts\activate
     ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Running the Project

- **To execute the full pipeline automatically:**
  ```sh
  python src/src.py
  ```
  This runs the entire workflow end-to-end without manual intervention.

- **To run and experiment with individual components:**
  1. Open Jupyter Notebook:
     ```sh
     jupyter notebook
     ```
  2. Run each notebook. The notebooks are named to match their corresponding scripts for easy navigation.
  3. You can also run each function manually to inspect intermediate results.

---

## Prerequisites

Ensure you have the following installed:

- Python (minimum version **3.8.10**)
- pip
- Git
- VS Code (or another preferred IDE)

---

## Dataset

The dataset used in this project is:

- **File:** `BrentOilPrices.csv`
- **Size:** 9,011 rows × 2 columns

---

## Project Requirements

### 1. GitHub Actions and Python Coding Standards

- **Automated code checks**: The repository includes **Pylint linters** in `.github/workflows` for maintaining coding standards.
- **Pull Request validation**: The linter runs automatically when a pull request is created.

#### Manual Linting Commands

- Check code formatting in the `scripts/` directory:
  ```sh
  pylint scripts/*.py
  ```

- Auto-fix linter errors in the `scripts/` directory:
  ```sh
  autopep8 --in-place --aggressive --aggressive scripts/*.py
  ```

- Check code formatting in the `src/` directory (main processing pipeline):
  ```sh
  pylint src/src.py
  ```

- Auto-fix linter errors in the `src/` directory:
  ```sh
  autopep8 --in-place --aggressive --aggressive src/src.py
  ```

---



### 2. Data Cleaning, Statistical Analysis, and EDA
This section is handled by the `BrentDataAnalysis` class in `data_analysis.py`. It performs essential preprocessing and exploratory data analysis (EDA) on the Brent oil price dataset. The key steps include:

- **Loading Data**: Reads the dataset from a CSV file, parsing dates and setting them as the index.
- **Overview & Summary**: Displays the dataset's shape, data types, and statistical summary for numerical and categorical columns.
- **Missing Value Identification**: Checks for missing values in the dataset.
- **Duplicate Removal**: Identifies and removes duplicate records to ensure data integrity.
- **Saving Processed Data**: Stores the cleaned dataset in a new CSV file for further analysis.
- **Time Series Plot**: Visualizes oil price trends over time using Seaborn and Matplotlib, saving the plot in the project's `notebooks/plots` directory.

### 3. Statistical Modeling
The `StatisticalModel` class in `statistical_modeling.py` implements advanced statistical techniques to analyze Brent oil price trends and volatility. It includes:

- **Stationarity Test**: Uses the Augmented Dickey-Fuller (ADF) test to check if the time series is stationary.
- **Autocorrelation & Partial Autocorrelation**: Plots ACF and PACF to help determine AR and MA orders for time series modeling.
- **ARIMA Model**: Fits an AutoRegressive Integrated Moving Average (ARIMA) model to capture trends and seasonality in oil price movements.
- **GARCH Model**: Fits a Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model to model price volatility.
- **Bayesian Inference**: Uses PyMC to perform probabilistic modeling, estimating uncertainty in oil price predictions.

Both sections work together within `src/src.py` to automate the end-to-end data processing, analysis, and modeling workflow.

The corresponding Jupyter notebook for `data_analysis.py` is `data_analyzer.ipynb`, and for `statistical_modeling.py`, it is `statistical_modeling.ipynb`. You can explore each method individually or run the entire script to gain insights, depending on your preferred usage of this repository.



### 4. Advanced Modeling (Change Point Analysis)
The `ChangePointAnalysis` class in `change_point_analysis.py` applies change point detection techniques to analyze shifts in Brent oil price trends due to economic and geopolitical factors. It includes:

- **Data Integration**: Merges historical Brent oil price data with economic indicators such as GDP, inflation, and exchange rates using data from FRED.
- **Scaling & Preprocessing**: Handles missing values and standardizes the dataset to improve model performance.
- **Vector Autoregression (VAR)**: Captures interdependencies among multiple time series variables to model economic influences on oil prices.
- **Markov Regime Switching Model**: Detects structural breaks and regime shifts in oil price trends using hidden Markov models.
- **Deep Learning (LSTM)**: Implements a Long Short-Term Memory (LSTM) model for time series forecasting, leveraging past trends for predictive insights.

This module provides insights into oil price trend shifts by integrating traditional econometric models with deep learning approaches. The corresponding Jupyter notebook for `change_point_analysis.py` is `change_point_analysis.ipynb`, allowing interactive exploration of its methodologies.



### 5. Developing an Interactive dashboard for Data Analysis


This section of this project is a **web-based dashboard** for analyzing Brent oil prices using statistical and deep learning models. The backend is built with **Flask**, while the frontend is developed using **React.js**. The system provides insights through **ARIMA, GARCH, Bayesian Inference, and LSTM models**.

## Features
- 📊 **Data Overview**: View and summarize Brent oil price data.
- 📉 **ARIMA Model**: Fetch and display ARIMA model results.
- 🔬 **Statistical Summary**: Provides key statistical insights.
- 🔄 **Model Comparison**: Evaluate and compare multiple forecasting models.

## Tech Stack
- **Backend**: Flask, Pandas, Scikit-learn, Statsmodels, LSTM (TensorFlow/Keras)
- **Frontend**: React.js, Axios, React Router
- **APIs**: RESTful API endpoints for data retrieval and model results

## Setup & Installation
### 1️⃣ Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```
### 2️⃣ Frontend
```bash
cd frontend
npm install
npm start
```

## API Endpoints
| Endpoint | Description |
|----------|------------|
| `/api/data` | Fetch oil price data |
| `/api/describe` | Get statistical summary |
| `/api/arima` | Retrieve ARIMA model results |
| `/api/compare_models` | Compare forecasting models |

## Usage
- Run the **backend** and **frontend** separately.
- Open `http://localhost:5000/` to access the backend endpoints and view data (You can leave or open based on your preferene)
- Open `http://localhost:3000/` to access the dashboard.




### More information
- You can refer to [this link]() to gain more insights about the reports of this project results.

## Authors

👤 **Amare Kassa**

- GitHub: [@githubhandle](https://github.com/amare1990)
- Twitter: [@twitterhandle](https://twitter.com/@amaremek)
- LinkedIn: [@linkedInHandle](https://www.linkedin.com/in/amaremek/)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

Feel free to check the [issues page](https://github.com/amare1990/brent-data-analysis/issues).

## Show your support

Give a ⭐️ if you like this project, and you are welcome to contribute to this project!

## Acknowledgments

- Hat tip to anyone whose code was referenced to.
- Thanks to the 10 academy and Kifiya financial instituion that gives me an opportunity to do this project

## 📝 License

This project is [MIT](./LICENSE) licensed.
