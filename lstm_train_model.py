import os # Python Standard Libraries
import json # Python Standard Libraries
import numpy as np # Numerical Computation
import tensorflow as tf  # TensorFlow / Keras: Deep Learning Framework
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts #Optimizer Tools
from tensorflow.keras.optimizers import Adam #Optimizer Tools
from tensorflow.keras.layers import LayerNormalization # Model Architecture Layers
from tensorflow.keras import layers, models # Model Architecture Layers
from tensorflow.keras.layers import Bidirectional # Model Architecture Layers
from tensorflow.keras.callbacks import ReduceLROnPlateau # Model Saving and Callbacks
from tensorflow.keras.models import load_model # Model Saving and Callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # Model Saving and Callbacks
from tensorflow.keras.utils import to_categorical # One-Hot Encoding
from sklearn.utils import class_weight # Classification Imbalance
import matplotlib.pyplot as plt # Plotting and Visualization
import matplotlib # Plotting and Visualization
matplotlib.use('Agg')  # Use non-interactive backend


# Paths
SPLITS_PATH = "D:\sdp project\splits"
MODEL_PATH = "lstm_trained_model.keras"
BEST_MODEL_PATH = "best_lstm_model.keras"
BATCH_SIZE = 16
EPOCHS = 60

# Load split datasets
def load_split_dataset(split_name):
    split_dir = os.path.join(SPLITS_PATH, split_name)
    X = np.load(os.path.join(split_dir, f"X_{split_name}.npy"))
    y = np.load(os.path.join(split_dir, f"y_{split_name}.npy"))
    return X, y

# Load datasets
X_train, y_train = load_split_dataset("train")
X_val, y_val = load_split_dataset("val")
X_test, y_test = load_split_dataset("test")

# Normalize input features
max_val = np.max([np.max(X_train), np.max(X_val), np.max(X_test)])
X_train = X_train / max_val
X_val = X_val / max_val
X_test = X_test / max_val

# Define augmentation function
def augment_sequence_data(X, noise_std):
    noise = np.random.normal(0, noise_std, X.shape)
    return X + noise

# Apply two levels of augmentation to triple X_train
X_aug1 = augment_sequence_data(X_train, noise_std=0.01)
X_aug2 = augment_sequence_data(X_train, noise_std=0.02)

# Combine original + 2 noisy versions
X_train = np.concatenate([X_train, X_aug1, X_aug2], axis=0)
y_train = np.concatenate([y_train, y_train, y_train], axis=0)

print("Tripled X_train with two rounds of Gaussian noise augmentation")

# Ensure labels are one-hot encoded
if len(y_train.shape) == 1:
    print("Labels are not one-hot encoded. Converting...")
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    y_test = to_categorical(y_test)

# Convert one-hot to label indices
y_train_labels = np.argmax(y_train, axis=1)

# Compute balanced class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)

# Convert to dictionary format expected by Keras
class_weights = dict(enumerate(class_weights))

print("Computed class weights:", class_weights)

NUM_CLASSES = y_train.shape[1]

print(f"Loaded splits: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

# Build the LSTM model

def build_lstm_model(sequence_length, num_classes):
    lr_schedule = CosineDecayRestarts(
        initial_learning_rate=0.0002,
        first_decay_steps=10,
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-5
    )

    model = models.Sequential([
        layers.Input(shape=(sequence_length, 63)),
        layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=2),
        Bidirectional(layers.LSTM(128, return_sequences=True)),
        LayerNormalization(),
        Bidirectional(layers.LSTM(64)),
        LayerNormalization(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=lr_schedule),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Define callbacks
callbacks = [
    EarlyStopping(patience=8, restore_best_weights=True),  # was 5
    ModelCheckpoint(BEST_MODEL_PATH, save_best_only=True),
]

# Train or Load model
if os.path.exists(MODEL_PATH):
    print("Pre-trained LSTM model found. Loading model...")
    model = load_model(MODEL_PATH)
    
    # Try to load saved history
    history = None
    if os.path.exists("lstm_training_history.json"):
        with open("lstm_training_history.json", "r") as f:
            history_data = json.load(f)
            history = type('History', (object,), {'history': history_data})()
        print("Training history loaded.")
else:
    print("No pre-trained model found. Training new model...")
    model = build_lstm_model(SEQUENCE_LENGTH, NUM_CLASSES)
    model.summary()

    history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    class_weight=class_weights  # Add this line
    )

    model.save(MODEL_PATH)
    print("Model trained and saved!")
    
    # Save training history
    with open("lstm_training_history.json", "w") as f:
        json.dump(history.history, f)
    print("Training history saved to lstm_training_history.json")

# Plot Accuracy and Loss Curves (if history is available)
if history is not None:
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train Accuracy')
    plt.plot(epochs_range, val_acc, label='Val Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("Accuracy and Loss Curves.png")
    print("Accuracy and Loss Curves saved as Accuracy and Loss Curves.png")
else:
    print("No training history available to plot Accuracy and Loss Curves.")

# Evaluate on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Evaluate on train data
train_loss, train_acc = model.evaluate(X_train, y_train)
print(f"Train Accuracy: {train_acc * 100:.2f}%")

# Classification Report (Precision, Recall, F1)
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Get predicted class labels
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n Classification Report:")
print(classification_report(y_true, y_pred, target_names=[f"Class {i}" for i in range(NUM_CLASSES)]))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"Class {i}" for i in range(NUM_CLASSES)])
plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Confusion Matrix saved as confusion_matrix.png")

# Multiclass ROC Curve
from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(NUM_CLASSES):
    # Make sure y_test is one-hot encoded; skip classes not in test set
    if np.sum(y_test[:, i]) == 0:
        print(f" Skipping ROC for Class {i} (not present in test set)")
        continue
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(NUM_CLASSES):
    if i in roc_auc:
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
print("ROC curve saved as roc_curve.png")