import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

file_path = '/content/dataset.csv'
data = pd.read_csv(file_path)

data = data[['N', 'P', 'K', 'label']]

# print("Missing values:\n", data.isnull().sum())

X = data[['N', 'P', 'K']]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy * 100:.2f}%")
# print("\nClassification Report:\n", classification_report(y_test, y_pred))

fertilizer_dict = {
    'grapes': 'Urea (N), DAP (P), MOP (K)',
    'mango': 'Urea (N), Super Phosphate (P), Potash (K)',
    'mulberry': 'Urea (N), DAP (P), MOP (K)',
    'potato': 'Ammonium Nitrate (N), Super Phosphate (P), Potash (K)',
    'ragi': 'Ammonium Sulphate (N), Rock Phosphate (P), MOP (K)',
    'pomegranate': 'Urea (N), DAP (P), MOP (K)'
}

def recommend_fertilizer(crop):
    return fertilizer_dict.get(crop, "Fertilizer information not available")

def recommend_crop_and_fertilizer(nitrogen, phosphorus, potassium):
    crop_prediction = rf_model.predict([[nitrogen, phosphorus, potassium]])[0]
    recommended_fertilizer = recommend_fertilizer(crop_prediction)
    return crop_prediction, recommended_fertilizer

nitrogen = int(input("Enter the value of Nitrogen: "))
phosphorus = int(input("Enter the value of Phosphorus: "))
potassium = int(input("Enter the value of Potassium: "))

recommended_crop, recommended_fertilizer = recommend_crop_and_fertilizer(nitrogen, phosphorus, potassium)
print(f"Recommended crop for N={nitrogen}, P={phosphorus}, K={potassium}: {recommended_crop}")
print(f"Fertilizer required: {recommended_fertilizer}")
