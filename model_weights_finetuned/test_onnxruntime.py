import onnx
import onnxruntime as ort
import numpy as np


"""
This script loads a DINs model in ONNX format, verifies its integrity, 
and performs inference using ONNX Runtime.
"""
# Display available execution providers
print("Available Execution Providers:", ort.get_available_providers())

# Path to the ONNX model (update accordingly)
onnx_model_path = "DINs/fold_1/checkpoint.onnx"

# Load the ONNX model and verify its integrity
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid.")

# Create an ONNX Runtime inference session
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
ort_session = ort.InferenceSession(onnx_model_path, providers=providers)

# Define the test input with expected shape (batch_size, depth, height, width, channels)
batch_size = 1
depth = 32 
height = 960
width = 320
channels_image = 1
channels_guide = 2

# Generate dummy test data
image_input = np.random.rand(batch_size, depth, height, width, channels_image).astype(np.float32)
guide_input = np.random.rand(batch_size, depth, height, width, channels_guide).astype(np.float32)

# Prepare input dictionary for inference
inputs = {
    "image": image_input,
    "guide": guide_input
}

# Run inference
outputs = ort_session.run(None, inputs)

# Print output shape and results
print("Output shape:", outputs[0].shape)
print("Sample output values:", outputs[0][0, 0, :5, :5, 0])  # Print a small slice of output
