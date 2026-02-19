import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ---------------- LOAD DATA ----------------
data = pd.read_csv("C:/Users/Lenovo/Desktop/online payment fraud detection/dataset/fraud_dataset.csv")  # change filename if needed

# ---------------- TARGET ----------------
if 'isFraud' not in data.columns:
    raise Exception("❌ 'isFraud' column not found in dataset")

y = data['isFraud']

# ---------------- FEATURES ----------------
X = data.drop('isFraud', axis=1)

# KEEP ONLY NUMERIC COLUMNS (KEY FIX)
X = X.select_dtypes(include=['int64', 'float64'])

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- SCALE ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ---------------- MODEL ----------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ---------------- SAVE MODEL ----------------
with open("payments.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ payments.pkl and scaler.pkl created successfully")
import pickle

model = pickle.load(open("payments.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
