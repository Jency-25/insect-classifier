import os
import sys

print("Starting TFLite conversion...")

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    print("TensorFlow not found. Please install it: pip install tensorflow")
    sys.exit(1)

MODEL_PATH = "model/insect_model.keras"
OUTPUT_PATH = "model/insect_model.tflite"

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model not found at {MODEL_PATH}")
    sys.exit(1)

print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimize for size and speed
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open(OUTPUT_PATH, 'wb') as f:
    f.write(tflite_model)

original_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
new_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)

print(f"\n✅ Conversion complete!")
print(f"   Original model size: {original_size:.1f} MB")
print(f"   TFLite model size:   {new_size:.1f} MB")
print(f"   Saved to: {OUTPUT_PATH}")
