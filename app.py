import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import MLPVAE  # Ensure this model is properly imported

# Load the trained model
model = torch.load("mlp_vae.pth", weights_only=False)
model.eval()  # Set to evaluation mode

# Define state names
state_names = ['temperature', 'map_line', 'map_cuff', 'pulse', 'unassisted_resp_rate',
               'end_tidal_co2', 'o2_flow_rate', 'base_excess', 'bicarb_(hco3)',
               'blood_urea_nitrogen_(bun)', 'creatinine', 'phosphorus', 'hemoglobin',
               'met_hgb', 'platelets', 'white_blood_cell_count', 'carboxy_hgb',
               'alanine_aminotransferase_(alt)', 'ammonia',
               'aspartate_aminotransferase_(ast)', 'bilirubin_total', 'fibrinogen',
               'inr', 'lactate_dehydrogenase', 'lactic_acid',
               'partial_prothrombin_time_(ptt)', 'prealbumin', 'lipase',
               'b-type_natriuretic_peptide_(bnp)',
               'partial_pressure_of_carbon_dioxide_(paco2)', 'ph',
               'saturation_of_oxygen_(sao2)', 'procalcitonin',
               'erythrocyte_sedimentation_rate_(esr)', 'gcs_total_score', 'best_map',
               'pf_sp', 'pf_pa', 'spo2', 'partial_pressure_of_oxygen_(pao2)',
               'rass_score', 'CAM_ICU']

# Function to generate synthetic data
def generate_synthetic_data(model, sample, modified_values):
    """
    Modifies multiple columns in the input and generates a synthetic output.
    """
    model.eval()
    sample = sample.clone().unsqueeze(0)  # Shape (1, input_dim)
    
    # Encode to latent space
    with torch.no_grad():
        mu, logvar = model.encode(sample)
        z = model.reparameterize(mu, logvar)
    
    # Modify the selected columns in the input
    modified_sample = sample.clone()
    for column_idx, new_value in modified_values.items():
        modified_sample[0, column_idx] = new_value
    
    # Re-encode after modification
    with torch.no_grad():
        new_mu, new_logvar = model.encode(modified_sample)
        new_z = model.reparameterize(new_mu, new_logvar)
    
    # Decode back to see changes
    generated_output = model.decode(new_z)
    return generated_output.squeeze().detach().numpy()  # Convert to NumPy for visualization

# Streamlit App UI
st.title("Interactive MLPVAE State Modifier")

# Load sample input (Replace with real data later)
train_tensor = torch.randn(1, 42)  # Dummy data (replace with real dataset input)
sample_data = train_tensor[0]

# Select multiple columns to modify
selected_columns = st.multiselect("Select States to Modify", options=range(42), format_func=lambda x: state_names[x])

# Dictionary to store new values
modified_values = {}
for column_idx in selected_columns:
    col1, col2 = st.columns([2, 1])
    with col1:
        new_value = st.slider(f"Set New Value for {state_names[column_idx]}", min_value=-5.0, max_value=5.0)
    with col2:
        typed_value = st.number_input(f"Type Value for {state_names[column_idx]}", value=float(new_value), key=f"num_{column_idx}")
    modified_values[column_idx] = typed_value

# Generate synthetic output
synthetic_output = generate_synthetic_data(model, sample_data, modified_values)

# Display original vs. modified data
st.subheader("Original vs. Modified Output")

fig, ax = plt.subplots()
ax.plot(sample_data.numpy(), label="Original", marker="o")
ax.plot(synthetic_output, label="Modified", marker="x")
ax.set_xticks(range(42))
ax.set_xticklabels(state_names, rotation=90)
ax.legend()
st.pyplot(fig)

st.write("Modified Output Values:", synthetic_output)
