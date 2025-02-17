import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = torch.load("mlp_vae.pth", map_location=torch.device("cpu"))
model.eval()  # Set to evaluation mode

# Function to generate synthetic data
def generate_synthetic_data(model, sample, column_idx, new_value):
    """
    Modifies a specific column in the input and generates a synthetic output.
    """
    model.eval()
    
    # Convert to batch format (1 sample)
    sample = sample.clone().unsqueeze(0)  # Shape (1, input_dim)
    
    # Encode to latent space
    with torch.no_grad():
        mu, logvar = model.encode(sample)
        z = model.reparameterize(mu, logvar)

    # Modify the column in the input
    modified_sample = sample.clone()
    modified_sample[0, column_idx] = new_value  # Modify column

    # Re-encode after modification
    with torch.no_grad():
        new_mu, new_logvar = model.encode(modified_sample)
        new_z = model.reparameterize(new_mu, new_logvar)

    # Decode back to see changes
    generated_output = model.decode(new_z)

    return generated_output.squeeze().detach().numpy()  # Convert to NumPy for visualization

# Streamlit App UI
st.title("Interactive MLPVAE Generator")

# Load sample input
sample_idx = 0  # Pick any row from dataset
train_tensor = torch.randn(10, 10)  # Dummy data (Replace with actual data)
sample_data = train_tensor[sample_idx]  # Select a sample input

# Select column to modify
column_to_change = st.slider("Select Column to Modify", min_value=0, max_value=sample_data.shape[0]-1, value=5)

# Set new value
new_value = st.slider("Set New Value", min_value=-5.0, max_value=5.0, value=float(sample_data[column_to_change]), step=0.1)

# Generate synthetic output
synthetic_output = generate_synthetic_data(model, sample_data, column_to_change, new_value)

# Display original vs. generated data
st.subheader("Original vs. Modified Output")

fig, ax = plt.subplots()
ax.plot(sample_data.numpy(), label="Original", marker="o")
ax.plot(synthetic_output, label="Modified", marker="x")
ax.legend()
st.pyplot(fig)

st.write("Modified Output Values:", synthetic_output)
