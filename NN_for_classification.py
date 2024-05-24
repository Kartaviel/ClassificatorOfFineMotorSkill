import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load training data
training_data = pd.read_csv('training_set.csv')
# Load research data
research_data = pd.read_csv('exp_set.csv')

X_train = training_data.iloc[:, :-1]  # Features (coordinates)
y_train = training_data.iloc[:, -1]   # Target labels (0 or 1). Обратить внимание, что в обучающем датасете должны быть как примеры "единицы" - показателей здоровых животных, так и "нули" - показатели больных. Тогда он будет разделять точки на группы

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Create a Multi-Layer Perceptron (MLP) classifier
model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=1000, random_state=42)
# Fit the model on the training data
model.fit(X_train_scaled, y_train)

# Prepare the research data
X_research = research_data.iloc[:, :-1]  # Features (coordinates)
X_research_scaled = scaler.transform(X_research)
# Predict the groups for the research data
research_predictions = model.predict(X_research_scaled)

research_data['Predicted_Group'] = research_predictions
research_data.to_csv('res_data.csv', index=False)
