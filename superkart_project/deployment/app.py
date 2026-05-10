import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib


# Download the model from the Model Hub
#model_path = hf_hub_download(repo_id="cjigar/superkart-package-model", filename="best_superkart_package_model_v1.joblib")

model_path = hf_hub_download(
    repo_id="cjigar/superkart-package-model", 
    filename="superkart_sales_model.joblib" 
)

# Load the model
model = joblib.load(model_path)

# Streamlit UI Setup
st.title("SuperKart Sales Forecast")
st.write("Enter product and store details below to predict the **Total Sales Revenue**.")

# Collect user input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Product Details")
    Product_Weight = st.number_input("Product Weight", min_value=0.0, value=12.0)
    Product_MRP = st.number_input("Product MRP ($)", min_value=0.0, value=150.0)
    Product_Allocated_Area = st.slider("Product Allocated Area Ratio", 0.0, 1.0, 0.05)
    Product_Sugar_Content = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    Product_Type = st.selectbox("Product Category", [
        "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household",
        "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast",
        "Health and Hygiene", "Hard Drinks", "Canned", "Breads", "Starchy Foods", "Others", "Seafood"
    ])
    Product_Id = st.text_input("Product ID", "FDX20")

with col2:
    st.subheader("Store Details")
    Store_Id = st.text_input("Store ID", "OUT049")
    Store_Establishment_Year = st.number_input("Establishment Year", min_value=1980, max_value=2026, value=1999)
    Store_Size = st.selectbox("Store Size", ["High", "Medium", "Small"])
    Store_Location_City_Type = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
    Store_Type = st.selectbox("Store Type", [
        "Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"
    ])

# ----------------------------
# Prepare input data
# ----------------------------
input_dict = {
    'Product_Weight': Product_Weight,
    'Product_Allocated_Area': Product_Allocated_Area,
    'Product_MRP': Product_MRP,
    'Store_Establishment_Year': Store_Establishment_Year,
    'Product_Id': Product_Id,
    'Product_Sugar_Content': Product_Sugar_Content,
    'Product_Type': Product_Type,
    'Store_Id': Store_Id,
    'Store_Size': Store_Size,
    'Store_Location_City_Type': Store_Location_City_Type,
    'Store_Type': Store_Type
}

input_df = pd.DataFrame([input_dict])

# Prediction Logic
if st.button("Predict Total Sales"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"### Estimated Total Sales: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
