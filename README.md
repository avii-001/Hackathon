Streamlit link: https://agricultural-yield-analysis.streamlit.app/

# 🌱 Agricultural Yield Analysis Dashboard

An interactive Streamlit-based dashboard that visualizes, analyzes, and predicts agricultural yield based on key environmental and farming factors. This project blends data analysis, machine learning, and user-friendly UI for smart agricultural decision-making.

## 🚀 Features

- 📊 **Data Exploration**: Visualize relationships between yield and key variables like fertilizer, irrigation, rainfall, and sunny days.
- 🔎 **Interactive Visualizations**: Built with Plotly for responsive, clean charts.
- 🧠 **Predictive Modeling**: Implements an OLS regression model to estimate crop yield.
- 📆 **Seasonal Prediction Tool**: Users can simulate yield outcomes based on selected Nepali months, accounting for average sunny days and rainfall.
- 🧹 **Data Cleaning**: Automatically handles outliers using IQR-based capping.
- 🧪 **Feature Engineering**: Categorizes irrigation and fertilizer levels into intuitive buckets (Low/Medium/High).

## 🗃️ Dataset

- File: `agricultural_yield_train.csv`
- Columns include:
  - `Seed_Variety`
  - `Irrigation_Schedule`
  - `Soil_Quality`
  - `Fertilizer_Amount_kg_per_hectare`
  - `Sunny_Days`
  - `Rainfall_mm`
  - `Yield_kg_per_hectare`

## 🛠️ Technologies Used

- [Streamlit](https://streamlit.io/) – for building the dashboard UI
- [Plotly](https://plotly.com/python/) – for rich, interactive visualizations
- [Pandas & NumPy](https://pandas.pydata.org/) – for data manipulation
- [Statsmodels](https://www.statsmodels.org/) – for OLS regression modeling
- [Scikit-learn](https://scikit-learn.org/) – for model evaluation (train/test split, metrics)
- [Seaborn & Matplotlib](https://seaborn.pydata.org/) – (optional) for advanced visualizations

## 🧪 Model Evaluation

- **RMSE**: Root Mean Squared Error
- **R² Score**: Coefficient of Determination
- The model helps predict `Yield_kg_per_hectare` using environmental and agricultural inputs.

## Screenshots
<img width="262" alt="image" src="https://github.com/user-attachments/assets/5f64e914-27c8-4ec3-8e71-4ddcdeedb862" />
<img width="263" alt="image" src="https://github.com/user-attachments/assets/034b427a-3a9c-45ae-b31c-e78212e0321c" />
<img width="263" alt="image" src="https://github.com/user-attachments/assets/1447ca6e-ec8f-464d-81a2-ae2edb66fafe" />
<img width="274" alt="image" src="https://github.com/user-attachments/assets/2e94f5e6-fe4a-4f98-818a-dc0739e1ab60" />

📌 Insights Derived
- Fertilizer and irrigation levels significantly influence yield.
- Sunny days and rainfall patterns are strong yield predictors.
- Predictive modeling enables actionable, data-driven decisions in agriculture.

📈 Future Improvements
- Integrate real-time weather data using APIs.
- Use more advanced regression models or ensemble techniques.
- Add farmer-level recommendations for best practices.

👩‍💻 Author
Kusumm Maharjan - https://github.com/avii-001
Kushpun Balami
Aliza Adhikari
Austin Karki
Narayan
