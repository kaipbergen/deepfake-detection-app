# deepfake_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# ---------------------------
# 1. Setup
# ---------------------------
data_dir = "data"  # should contain training_real/ and training_fake/
target_size = (224, 224)
batch_size = 32

# ---------------------------
# 2. Data Generators
# ---------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1,
)

print("Loading training data...")
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="training",
    color_mode="rgb"
)

print("Loading validation data...")
validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
    color_mode="rgb"
)

print(f"\nTraining samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Classes: {train_generator.class_indices}")

# ---------------------------
# 3. Build Model (Xception Transfer Learning)
# ---------------------------
base_model = Xception(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model initially
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

model.summary()

# ---------------------------
# 4. Callbacks
# ---------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("best_deepfake_model.keras", save_best_only=True, monitor="val_accuracy"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6)
]

# ---------------------------
# 5. Training - Stage 1 (Top Layers Only)
# ---------------------------
print("\n--- Stage 1: Training Top Layers ---")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=callbacks
)

# ---------------------------
# 6. Fine-tuning Entire Model
# ---------------------------
print("\n--- Stage 2: Fine-tuning Full Model ---")
base_model.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

history_fine = model.fit(
    train_generator,
    epochs=20,
    validation_data=validation_generator,
    callbacks=callbacks
)

# ---------------------------
# 7. Save Final Model
# ---------------------------
model.save("deepfake_final_model.keras")
print("\n✅ Training complete. Model saved as 'deepfake_final_model.keras'")

# ---------------------------
# 8. Plot Training History
# ---------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"] + history_fine.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"] + history_fine.history["val_accuracy"], label="Val Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training & Validation Accuracy")
plt.show()