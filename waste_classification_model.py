# Waste Classification using MobileNetV2 + Confusion Matrix + Accuracy/Loss Graphs
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# ----------------------------
# Paths
# ----------------------------
train_dir = r"C:\Users\asus\OneDrive\Desktop\Projects\AI\Sustainability\dataset\dataset\train"
test_dir = r"C:\Users\asus\OneDrive\Desktop\Projects\AI\Sustainability\dataset\dataset\test"

# Image settings
img_height, img_width = 150, 150
batch_size = 32

# ----------------------------
# Data preprocessing
# ----------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ----------------------------
# Build Transfer Learning Model
# ----------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ----------------------------
# Train model
# ----------------------------
history = model.fit(
    train_data,
    epochs=15,
    validation_data=test_data
)

# ----------------------------
# Evaluate model
# ----------------------------
loss, acc = model.evaluate(test_data)
print(f"\n✅ Test Accuracy: {acc*100:.2f}%")

# ----------------------------
# Confusion Matrix + Classification Report
# ----------------------------
print("\n📊 Generating Confusion Matrix and Classification Report...")

y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_data.classes
class_labels = list(test_data.class_indices.keys())

# Check for missing predicted classes
predicted_classes = np.unique(y_pred_classes)
print(f"\n🔍 Predicted classes found: {predicted_classes}")
if len(predicted_classes) < len(class_labels):
    missing = set(np.arange(len(class_labels))) - set(predicted_classes)
    print(f"⚠️ Missing predicted classes: {[class_labels[i] for i in missing]}")
else:
    print("✅ All classes were predicted at least once.")

# Confusion Matrix (show all labels even if some not predicted)
cm = confusion_matrix(y_true, y_pred_classes, labels=np.arange(len(class_labels)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", bbox_inches='tight')
print("📁 Saved confusion matrix as confusion_matrix.png")
plt.show()

# Classification report (handles missing classes safely)
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred_classes,
    labels=np.arange(len(class_labels)),
    target_names=class_labels,
    zero_division=0
))

# ----------------------------
# Save model
# ----------------------------
model.save("waste_classification_mobilenetv2.h5")
print("\n💾 Model saved as waste_classification_mobilenetv2.h5")

# ----------------------------
# Plot Accuracy & Loss Graphs
# ----------------------------
plt.figure(figsize=(10, 4))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy (MobileNetV2)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss (MobileNetV2)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_results_mobilenetv2.png", bbox_inches='tight')
print("📁 Saved training graphs as training_results_mobilenetv2.png")
plt.show()
