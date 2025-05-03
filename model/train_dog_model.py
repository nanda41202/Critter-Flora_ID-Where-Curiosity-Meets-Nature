import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define paths
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(MODEL_DIR, 'dog_images', 'train')
VAL_DIR = os.path.join(MODEL_DIR, 'dog_images', 'val')
MODEL_PATH = os.path.join(MODEL_DIR, 'dog_breed_classifier.h5')

# Check if training data exists
if not os.path.exists(TRAIN_DIR):
    print(f"Training directory not found at {TRAIN_DIR}")
    print("Please create a directory structure with training images:")
    print("model/dog_images/train/<breed_name>/<images>")
    print("model/dog_images/val/<breed_name>/<images>")
    exit(1)

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load the base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size
)

# Save the model
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Save class names
with open(os.path.join(MODEL_DIR, 'dog_class_names.txt'), 'w') as f:
    for class_name in train_generator.class_indices:
        f.write(f"{class_name}\n")
print("Class names saved to dog_class_names.txt") 