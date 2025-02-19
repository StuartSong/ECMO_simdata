import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import MLPVAE  # Ensure model definition is available
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model properly
device = torch.device("cpu")  # Ensure compatibility with Streamlit
input_dim = 42  # Adjust based on your actual input dimensions
model = MLPVAE(input_dim=input_dim)  # Initialize model
model.load_state_dict(torch.load("mlp_vae.pth", map_location=device))  # Load state_dict
model.to(device)
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
    model.eval()  # Ensure model is in eval mode
    sample = sample.clone().unsqueeze(0).to(device)  # Ensure batch size (1, input_dim)
    
    with torch.no_grad():
        mu, logvar = model.encode(sample)
        z = model.reparameterize(mu, logvar)
    
    modified_sample = sample.clone()
    for column_idx, new_value in modified_values.items():
        modified_sample[0, column_idx] = new_value  # Modify selected columns
    
    with torch.no_grad():
        new_mu, new_logvar = model.encode(modified_sample)
        new_z = model.reparameterize(new_mu, new_logvar)
        generated_output = model.decode(new_z)
    
    return generated_output.squeeze().detach().cpu().numpy()  # Convert to NumPy for visualization

# Streamlit App UI
st.title("Interactive MLPVAE State Modifier")

# Load and preprocess test data
test_data = pd.read_csv("non_discritized_states.csv", index_col=0)
if 'csn' in test_data.columns:
    test_data.drop(columns=['csn'], inplace=True)  # Ensure 'csn' is removed

scaler = StandardScaler()
test_data = scaler.fit_transform(test_data)

# Ensure test_data has correct shape
if test_data.shape[1] != input_dim:
    st.error(f"Expected input dimension {input_dim}, but found {test_data.shape[1]}")
    st.stop()

# Load sample input
sample_data = torch.tensor(test_data[0, :], dtype=torch.float32).to(device)  # Ensure correct shape

# Select multiple columns to modify
selected_columns = st.multiselect("Select States to Modify", options=range(len(state_names)), format_func=lambda x: state_names[x])

# Dictionary to store new values
modified_values = {}
for column_idx in selected_columns:
    col1, col2 = st.columns([2, 1])
    with col1:
        new_value = st.slider(f"Set New Value for {state_names[column_idx]}", 
                              min_value=float(test_data[:, column_idx].min()), 
                              max_value=float(test_data[:, column_idx].max()), 
                              value=float(test_data[0, column_idx]))
    with col2:
        typed_value = st.number_input(f"Type Value for {state_names[column_idx]}", value=float(new_value), key=f"num_{column_idx}")
    modified_values[column_idx] = typed_value

# Generate synthetic output
synthetic_output = generate_synthetic_data(model, sample_data, modified_values)

# Apply inverse transformation (undo scaling)
synthetic_output_rescaled = scaler.inverse_transform(synthetic_output.reshape(1, -1))
sample_data_rescaled = scaler.inverse_transform(sample_data.cpu().numpy().reshape(1, -1))

# Display original vs. modified data
st.subheader("Original vs. Modified Output")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(sample_data_rescaled.flatten(), label="Original", marker="o", linestyle="dashed")
ax.plot(synthetic_output_rescaled.flatten(), label="Modified", marker="x", linestyle="solid")
ax.set_xticks(range(len(state_names)))
ax.set_xticklabels(state_names, rotation=90, fontsize=8)
ax.legend()
st.pyplot(fig)

st.write("Modified Output Values:")
st.dataframe(pd.DataFrame(synthetic_output_rescaled, columns=state_names))
