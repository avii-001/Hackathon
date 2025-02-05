import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
st.title("🌱 Agricultural Yield Analysis Dashboard")
DATA_PATH = "agricultural_yield_train.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

df = load_data()

# Data Overview
st.subheader("📊 Dataset Overview")
st.write(df.head())

# Data Cleaning (Handling Outliers)
for col in ["Sunny_Days", "Rainfall_mm", "Irrigation_Schedule"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower_bound, upper_bound)

# Feature Engineering
df["Fertilizer"] = df["Fertilizer_Amount_kg_per_hectare"].apply(lambda x: "Less" if x < 100 else ("Medium" if x < 200 else "High"))
df["Irrigation"] = df["Irrigation_Schedule"].apply(lambda x: "Less" if x < 4 else ("Medium" if x < 7 else "High"))


# Interactive Visualizations
fig = px.scatter(df, x="Fertilizer_Amount_kg_per_hectare", y="Yield_kg_per_hectare",
                 #size="Sunny_Days", 
                 color="Seed_Variety", hover_data=df.columns,
                 title='🎈Fertilizer vs Yield by Seed Variety',
                 )
fig.update_layout(
    title_font=dict(size=30),  # Title font size (24 + default ~ 8)
    legend=dict(font=dict(size=24))  # Legend font size
)
st.plotly_chart(fig)





# ✅ Irrigation vs Yield (Boxplot)
fig2 = px.box(df, x="Irrigation_Schedule", y="Yield_kg_per_hectare", color="Irrigation_Schedule",
              title='💦 Irrigation Impact on Yield')
fig2.update_layout(title_font=dict(size=30))
st.plotly_chart(fig2)

# # ✅ Fertilizer vs Yield (Bar Chart)
# fig3 = px.bar(df, x="Fertilizer_Amount_kg_per_hectare", y="Yield_kg_per_hectare", 
#               title='🌱 Fertilizer Effect on Yield',color="Fertilizer_Amount_kg_per_hectare", barmode="group")
# fig3.update_layout(title_font=dict(size=30))
# st.plotly_chart(fig3)

fig4 = px.histogram(df, x="Rainfall_mm", y="Yield_kg_per_hectare", title='🌧️ Rainfall vs Yield')
fig4.update_layout(title_font=dict(size=30))
st.plotly_chart(fig4)


# Categorize 'Sunny_Days' into 5 quantile-based bins
df["Sunny_Days_Cat"] = pd.qcut(df["Sunny_Days"], q=5, labels=["Very Low", "Low", "Medium", "High", "Very High"])

# Calculate the average yield for each sunny days category
avg_yield_sunny = df.groupby("Sunny_Days_Cat")["Yield_kg_per_hectare"].mean().reset_index()

# Create a line plot
fig_sunny = px.line(avg_yield_sunny, x="Sunny_Days_Cat", y="Yield_kg_per_hectare", markers=True,
                    title="☀️ Average Yield Based on Sunny Days Category")

# Increase font sizes
fig_sunny.update_layout(
    title_font=dict(size=30),
    legend=dict(font=dict(size=24))
)

# Display the plot in Streamlit
st.plotly_chart(fig_sunny)

# Machine Learning Model (OLS Regression)
st.subheader("🧠 Yield Prediction Model")
X = df[["Seed_Variety", "Irrigation_Schedule", "Soil_Quality", "Fertilizer_Amount_kg_per_hectare", "Sunny_Days", "Rainfall_mm"]]
Y = df["Yield_kg_per_hectare"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=120)

# Fit Regression Model
X_train_sm = sm.add_constant(X_train)
lr = sm.OLS(Y_train, X_train_sm).fit()

# Show Model Summary
st.text(lr.summary())

# Predictions
X_test_sm = sm.add_constant(X_test)
y_pred = lr.predict(X_test_sm)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
r_squared = r2_score(Y_test, y_pred)

st.write(f"📉 **RMSE:** {rmse:.2f}")
st.write(f"📈 **R² Score:** {r_squared:.2f}")

# Conclusion
st.markdown("### ✅ Key Insights")
st.write("- Fertilizer and Irrigation significantly impact yield.")
st.write("- More Sunny Days contribute to higher yields.")
st.write("- The ML model can predict yield based on input parameters.")





import statsmodels.api as sm

# Define Nepali months and average values
nepali_months = {
    "Baisakh": 1, "Jestha": 2, "Ashadh": 3, "Shrawan": 4, "Bhadra": 5, "Ashwin": 6,
    "Kartik": 7, "Mangsir": 8, "Poush": 9, "Magh": 10, "Falgun": 11, "Chaitra": 12
}

# Average Sunny Days per Month
avg_sunny_days = {
    1: 26, 2: 27, 3: 18, 4: 12, 5: 15, 6: 22,
    7: 27, 8: 28, 9: 29, 10: 28, 11: 27, 12: 25
}

# Average Rainfall per Month (in mm)
avg_rainfall = {
    1: 20, 2: 30, 3: 120, 4: 200, 5: 300, 6: 250,
    7: 180, 8: 100, 9: 50, 10: 30, 11: 25, 12: 15
}

# Streamlit UI
st.title("🌱 Agricultural Yield Prediction")

# Dropdown for selecting month by name
selected_month_name = st.selectbox("Select Start Month", list(nepali_months.keys()))
input_month = nepali_months[selected_month_name]  # Convert selected month name to number

# Number input for duration
duration = st.number_input("Enter duration (in months)", min_value=1, max_value=12, value=3)

# Calculate total sunny days
total_sunny_days = 0
total_rainfall = 0
current_month = input_month

for _ in range(duration):
    total_sunny_days += avg_sunny_days[current_month]
    total_rainfall += avg_rainfall[current_month]  # Add rainfall for the month
    current_month = 1 if current_month == 12 else current_month + 1  # Cycle months

# Display results
st.write(f"☀️ **Total Sunny Days over {duration} months: {total_sunny_days}**")
st.write(f"🌧️ **Total Rainfall over {duration} months: {total_rainfall} mm**")

# Other Inputs
soil_quality = st.slider("Soil Quality", min_value=50, max_value=100)
seed_variety = st.selectbox("Seed Variety", options=[("Low Yield", 0), ("High Yield", 1)], format_func=lambda x: x[0])
seed_variety = seed_variety[1]  # Extract the numeric value (0 or 1)
fertilizer_amount = st.slider("Fertilizer Amount (kg per hectare)", min_value=0, max_value=500, value=150)
irrigation_schedule = st.slider("Irrigation Schedule", min_value=1, max_value=20, value=5)

fertilizer_level = 0 if fertilizer_amount < 100 else (1 if fertilizer_amount < 200 else 2)
irrigation_level = 0 if irrigation_schedule < 4 else (1 if irrigation_schedule < 7 else 2)

# Use calculated values for prediction
sunny_days = total_sunny_days
rainfall = total_rainfall

# Create input DataFrame
input_data = pd.DataFrame([[seed_variety, irrigation_schedule, soil_quality, fertilizer_amount, sunny_days, rainfall]],
                          columns=X.columns)

# Predict Yield
input_data = sm.add_constant(input_data, has_constant="add")
predicted_yield = lr.predict(input_data)[0]

st.write(f"### 🌾 Predicted Yield: **{predicted_yield:.2f} kg per hectare**")
