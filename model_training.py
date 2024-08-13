import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
data = pd.read_csv('hand_landmarks_data.csv')

# Prepare the data
X = data.drop('label', axis=1).values  # Hand landmarks
y = data['label'].values  # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode the labels
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# Define the model architecture
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # Input shape based on features
    Dense(64, activation='relu'),
    Dense(len(lb.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the model for later use
model.save('hand_sign_model.h5')
# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)

print(f'Test Accuracy: {accuracy*100:.2f}%')
