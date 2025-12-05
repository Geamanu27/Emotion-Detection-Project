import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


model_path = '../work_dirs/mobileNet(3 classes).h5'
data_dir = '../data/data_3c/val'
img_size = (224, 224)
batch_size = 32

print(f"Loading MobileNet from {model_path}...")

try:
    # 2. LOAD THE SAVED MODEL
    model = load_model(model_path)
    print("Model loaded successfully.")

    # 3. PREPARE DATA GENERATOR (MobileNet Specific)
    # IMPORTANT: We MUST rescale by 1./255 for MobileNet
    test_datagen = ImageDataGenerator(rescale=1./255)

    print("Loading Validation Data...")
    # Shuffle=False is critical for matching predictions to true labels!
    validation_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # 4. RUN PREDICTIONS
    print("Running inference... (This may take a minute)")
    # We use (samples // batch_size + 1) to ensure we cover all images
    Y_pred = model.predict(validation_generator, validation_generator.samples // batch_size + 1)
    y_pred = np.argmax(Y_pred, axis=1)

    # Get True Labels
    # We purposefully limit y_true to the length of y_pred to handle any batch size rounding
    y_true = validation_generator.classes[:len(y_pred)]
    class_labels = list(validation_generator.class_indices.keys())

    # 5. GENERATE CONFUSION MATRIX
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title('Confusion Matrix (MobileNet)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('mobilenet_confusion_matrix.png')
    plt.show()

    # 6. GENERATE ACCURACY BAR CHART
    # Calculate metrics for each class
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)

    plt.figure(figsize=(10, 6))
    # Using F1 Score as the metric for the bar chart
    sns.barplot(x=class_labels, y=f1, palette='viridis')
    plt.title('Performance by Emotion (F1 Score)')
    plt.ylim(0, 1.0)
    plt.ylabel('F1 Score')
    plt.xlabel('Emotion')
    plt.tight_layout()
    plt.savefig('mobilenet_accuracy_chart.png')
    plt.show()

    # 7. PRINT REPORT
    print("\n=== Classification Report ===")
    print(classification_report(y_true, y_pred, target_names=class_labels))

except Exception as e:
    print(f"\nError: {e}")
    print("Please check that your file path is correct and the .h5 file exists.")