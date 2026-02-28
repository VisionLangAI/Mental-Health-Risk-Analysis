
# psychological_wellbeing_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import shap
import lime
import lime.lime_tabular

# 1. Load Data
df = pd.read_csv('mental_health_data.csv')

# 2. Preprocessing
df.fillna(df.median(), inplace=True)
X = df.drop('mental_health_risk', axis=1)
y = df['mental_health_risk']

# 3. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Model Training - Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 6. Evaluation
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. Explainability - SHAP
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=X.columns)

# 8. Explainability - LIME
lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=X.columns.tolist(),
                                                        class_names=np.unique(y).astype(str), verbose=True, mode='classification')
exp = lime_explainer.explain_instance(X_test[0], clf.predict_proba, num_features=10)
exp.show_in_notebook(show_table=True)

# NOTE: This is a simplified example; in practice, integrate additional components such as CAAL Layer, FT-Transformer, etc.
