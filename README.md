# 🎮 Video Game Sales Analytics

An interactive web application for analyzing and predicting video game sales data for PS4 and Xbox One platforms. This project provides comprehensive analytics, visualizations, and machine learning-based sales predictions using a Random Forest model.

## 📋 Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Dashboard Features](#dashboard-features)
- [Machine Learning Model](#machine-learning-model)
- [Future Enhancements](#future-enhancements)

## ✨ Features

- **📊 Key Performance Metrics**: Total games, total global sales, and average sales per game
- **📈 Interactive Visualizations**: 
  - Platform-wise sales comparison (PS4 vs Xbox One)
  - Genre-wise sales analysis
  - Year-wise sales trends
  - Regional sales distribution (PS4 by country)
- **🔮 Sales Prediction**: Machine learning model to predict global sales based on platform and genre
- **🌍 Regional Analysis**: Detailed breakdown of sales by region (North America, Europe, Japan, Rest of World)
- **📱 Interactive Dashboard**: User-friendly Streamlit interface with real-time analytics

## 🛠 Technologies Used

- **Python 3.x**
- **Streamlit** - Web application framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Plotly** - Interactive visualizations
- **Scikit-learn** - Machine learning (Random Forest Regressor)
- **Jupyter Notebook** - Data cleaning and preprocessing
- **Power BI** - Additional analytics and reporting

## 📁 Project Structure

```
Video-Game-Sales-Analytics/
│
├── streamlit_app.py              # Main Streamlit application
├── Data_Clean.ipynb              # Data cleaning and preprocessing notebook
├── requirements.txt              # Python dependencies
│
├── Data Files:
│   ├── Video_Game.csv            # Combined and cleaned video game sales data
│   ├── PS4_GamesSales.csv        # Raw PS4 sales data
│   └── XboxOne_GameSales.csv     # Raw Xbox One sales data
│
└── README.md                      # Project documentation
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Video-Game-Sales-Analytics.git
   cd Video-Game-Sales-Analytics
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### Running the Streamlit App

1. **Start the application**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Access the dashboard**
   - The app will automatically open in your default web browser
   - If not, navigate to `http://localhost:8501`

### Dashboard Features

The interactive dashboard includes:

1. **Key Metrics Section**
   - Total number of games analyzed
   - Total global sales in millions
   - Average sales per game

2. **Sales Insights**
   - Platform-wise global sales comparison (PS4 vs Xbox One)
   - Genre-wise sales analysis with ranking
   - Year-wise sales trends with interactive line chart

3. **Regional Analysis**
   - PS4 sales distribution across regions:
     - North America
     - Europe
     - Japan
     - Rest of World

4. **Sales Prediction Simulator**
   - Select a platform (PS4 or Xbox One)
   - Choose a game genre
   - Get predicted global sales using the trained Random Forest model

### Using the Data Cleaning Notebook

1. **Open Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Run the Data_Clean.ipynb notebook**
   - Processes raw CSV files (`PS4_GamesSales.csv` and `XboxOne_GameSales.csv`)
   - Handles missing values, duplicates, and data standardization
   - Generates cleaned combined dataset (`Video_Game.csv`)

## 📊 Data Sources

The project uses combined video game sales data from:
- **PS4 Games**: 1,033 games
- **Xbox One Games**: 613 games
- **Total Dataset**: ~1,646 games (after cleaning and preprocessing)

### Data Fields

- **Game**: Name of the video game
- **Year**: Release year
- **Genre**: Game genre (Action, Adventure, RPG, Sports, etc.)
- **Publisher**: Game publisher
- **North America**: Sales in North America (million units)
- **Europe**: Sales in Europe (million units)
- **Japan**: Sales in Japan (million units)
- **Rest of World**: Sales in other regions (million units)
- **Global**: Total global sales (million units)
- **Platform**: Gaming platform (PS4 or Xbox One)

## 🎯 Dashboard Features

### 1. Key Business Metrics
- Real-time calculation of total games, sales, and averages
- Displayed in metric cards for quick insights

### 2. Interactive Charts
- **Bar Charts**: Platform and genre comparisons
- **Line Charts**: Temporal sales trends over years
- **Horizontal Bar Charts**: Genre performance rankings
- All charts are interactive and built with Plotly

### 3. Regional Sales Analysis
- Detailed breakdown of PS4 sales by region
- Color-coded bar chart for market distribution
- Quick identification of top-performing regions

### 4. Sales Prediction Model
- **Algorithm**: Random Forest Regressor
- **Features**: Platform and Genre
- **Training Data**: 70% of combined dataset
- **Output**: Predicted global sales in million units

## 🤖 Machine Learning Model

The project uses a **Random Forest Regressor** for sales prediction:

### Model Architecture
- **Algorithm**: Random Forest Regressor
- **Number of Estimators**: 300 trees
- **Preprocessing**: One-hot encoding for categorical features (Platform, Genre)
- **Random State**: 42 (for reproducibility)
- **Parallel Processing**: Enabled (n_jobs=-1)

### Model Workflow
1. Input features: Platform and Genre (categorical variables)
2. One-hot encoding converts categorical features to numerical format
3. Random Forest model trained on 70% of the dataset
4. Predictions generated for user-selected platform and genre combinations
5. Model is cached using Streamlit for optimal performance

### Output
- Predicted global sales in **million units**
- Results displayed in real-time on the dashboard

## 🔮 Future Enhancements

Potential improvements for the project:
- Add more platforms (Nintendo Switch, PC, etc.)
- Include additional predictive features (Publisher, Year, etc.)
- Implement model evaluation metrics (R², MAE, RMSE)
- Add data export/download functionality
- Create time series forecasting for future sales
- Add interactive filtering by region or publisher
- Implement user authentication for personalized dashboards
- Deploy to cloud platform (Heroku, AWS, Azure)

## 📝 Notes

- Ensure `Video_Game.csv` is in the root directory before running the Streamlit app
- The data cleaning notebook should be run first if processing raw data
- Missing values and duplicates are handled during data preprocessing
- The Streamlit app caches data and model training for improved performance

## 📄 License

This project is open source and available for educational and research purposes.

## 👤 Author

Created as a data analytics and machine learning project for video game sales analysis and prediction.

---

**Happy exploring! 🎮📊**
