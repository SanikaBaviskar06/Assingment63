import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    #values scale
from sklearn.neural_network import MLPClassifier    #neural network model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#-----------------------------------------------------------
# 1. Dataset load
#-----------------------------------------------------------
df = pd.read_csv("Loan_Default.csv")

print(df.head())
print(df.shape)

#-----------------------------------------------------------
# 2. Missing values
#-----------------------------------------------------------
print(df.isnull().sum())

df = df.fillna(df.median(numeric_only=True))

#-----------------------------------------------------------
# 3. Convert categorical columns
#-----------------------------------------------------------
df = pd.get_dummies(df, drop_first=True)

#-----------------------------------------------------------
# 4. X and y
#-----------------------------------------------------------
X = df.drop("Default", axis=1)
y = df["Default"]

#-----------------------------------------------------------
# 5. Train Test Split
#-----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#-----------------------------------------------------------
# 6. Scaling
#-----------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#-----------------------------------------------------------
# 7. MLP Model
#-----------------------------------------------------------
model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)

#-----------------------------------------------------------
# 8. Training
#-----------------------------------------------------------
model.fit(X_train, y_train)

#-----------------------------------------------------------
# 9. Prediction
#-----------------------------------------------------------
y_pred = model.predict(X_test)

#-----------------------------------------------------------
# 10. Results
#-----------------------------------------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#-----------------------------------------------------------
# 11. Loss graph
#-----------------------------------------------------------
plt.plot(model.loss_curve_)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

#-----------------------------------------------------------
# 12. New Applicant Prediction
#-----------------------------------------------------------

new_data = pd.DataFrame(0, index=[0], columns=X.columns)

new_data["Age"] = 30
new_data["Income"] = 600000
new_data["LoanAmount"] = 300000
new_data["CreditScore"] = 700
new_data["EmploymentYears"] = 5
new_data["ExistingLoans"] = 1
new_data["MonthlyDebt"] = 10000
new_data["LoanTerm"] = 36

new_data = scaler.transform(new_data)

result = model.predict(new_data)

print("New Applicant Prediction:", result[0])

if result[0] == 0:
    print("Low default risk")
else:
    print("High default risk")

#read_csv =data load karnyasathi, x=input, y=default

# Experiment 1 - Activation Function

for act in ["identity", "logistic", "tanh", "relu"]:

    model1 = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation=act,
        max_iter=500,
        random_state=42
    )

    model1.fit(X_train, y_train)

    pred = model1.predict(X_test)

    print(act, "Accuracy:", accuracy_score(y_test, pred))

# Experiment 2 - Hidden Layers

for layers in [(10,), (20, 10), (50, 25), (100, 50, 25)]:

    model2 = MLPClassifier(
        hidden_layer_sizes=layers,
        activation="relu",
        max_iter=500,
        random_state=42
    )

    model2.fit(X_train, y_train)

    pred = model2.predict(X_test)

    print(layers, "Accuracy:", accuracy_score(y_test, pred))

# Experiment 3 - Learning Rate

for rate in [0.001, 0.01, 0.1]:

    model3 = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        learning_rate_init=rate,
        max_iter=500,
        random_state=42
    )

    model3.fit(X_train, y_train)

    pred = model3.predict(X_test)

    print(rate, "Accuracy:", accuracy_score(y_test, pred))
