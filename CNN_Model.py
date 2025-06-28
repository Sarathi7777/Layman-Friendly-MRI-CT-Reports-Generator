import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Set paths
preprocessed_data_dir = "C:\\GOAT\\Processed_Output"  # Corrected path with double backslashes
categories = os.listdir(preprocessed_data_dir)  # Assuming subfolders are class labels

# Load data
def load_data(data_dir):
    images = []
    labels = []
    
    # Iterate through each category (subfolder in the data directory)
    for label, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        
        # Check if the category_path is a directory
        if not os.path.isdir(category_path):
            print(f"Skipping non-directory file: {category_path}")
            continue
        
        # Iterate through each file in the category directory
        for file in os.listdir(category_path):
            file_path = os.path.join(category_path, file)
            
            # Check if the file is a valid image file
            if os.path.isfile(file_path) and file.endswith(('png', 'jpg', 'jpeg')):
                image = tf.keras.utils.load_img(file_path, target_size=(224, 224))  # Load image and resize
                image = tf.keras.utils.img_to_array(image)  # Convert image to array
                images.append(image)
                labels.append(label)  # Assign label based on category
            else:
                print(f"Skipping non-image file: {file_path}")
    
    # Convert lists to tensors
    return tf.convert_to_tensor(images), tf.convert_to_tensor(labels)

# Load images and labels
images, labels = load_data(preprocessed_data_dir)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Normalize images
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')  # Output layer with one neuron per category
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=2,
    batch_size=32
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save the model
model.save("cnn_image_analysis_model.h5")
print("Model saved.")
