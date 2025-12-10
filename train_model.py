import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import Rescaling
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

train_dir = "Dataset/train"
valid_dir = "Dataset/valid"

# Load dataset
train_data = image_dataset_from_directory(train_dir, image_size=(256,256), batch_size=32)
val_data = image_dataset_from_directory(valid_dir, image_size=(256,256), batch_size=32)

# Get class names
class_names = val_data.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Normalize
rescale = Rescaling(1/255)
train_data = train_data.map(lambda x,y: (rescale(x),y))
val_data = val_data.map(lambda x,y: (rescale(x),y))

# CNN Model
model = keras.Sequential([
    keras.layers.Input(shape=(256,256,3)),
    keras.layers.Conv2D(32,(3,3),activation='relu',padding='same'),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Conv2D(64,(3,3),activation='relu',padding='same'),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Conv2D(128,(3,3),activation='relu',padding='same'),
    keras.layers.MaxPooling2D(2,2),

    keras.layers.Flatten(),
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes,activation='softmax')
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(train_data, validation_data=val_data, epochs=10)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/hibiscus_model.h5")
print("Model saved as models/hibiscus_model.h5")

# ---------------------------------------------
# CONFUSION MATRIX & REPORT
# ---------------------------------------------

# Collect predictions & labels
y_true = []
y_pred = []

for batch_images, batch_labels in val_data:
    preds = model.predict(batch_images)
    y_true.extend(batch_labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nCONFUSION MATRIX:\n", cm)

# Classification Report
print("\nCLASSIFICATION REPORT:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# Plot confusion matrix
plt.figure(figsize=(6,5))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

# Labels
for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i][j], ha='center', va='center', color='red')

plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

os.makedirs("static/results", exist_ok=True)
plt.savefig("static/results/confusion_matrix.png")
plt.show()
