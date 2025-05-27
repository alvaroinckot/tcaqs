#!/usr/bin/env python3
# Character Bid Prediction - Simple Gradio Web Interface
# This script creates a web interface for the trained XGBoost model
# that predicts character bid values based on their attributes.

import os
import sys
import pickle
import pandas as pd
import numpy as np
import datetime

# Add parent directory to path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Model path
MODEL_PATH = '../models/xgboost_character_bid_model.pkl'

# Load the model and preprocessor
def load_model(model_path=MODEL_PATH):
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            preprocessor = model_data['preprocessor']
        print("Model loaded successfully.")
        return model, preprocessor
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# Define vocations for dropdown
VOCATIONS = ['knight', 'paladin', 'sorcerer', 'druid']

# Define server suffixes for dropdown
SERVER_SUFFIXES = ['Optional-PvP', 'Open-PvP', 'Hardcore-PvP']

# Function to make predictions
def predict_character_bid(character_data, model, preprocessor):
    """Make bid prediction for character data."""
    # Add required columns if missing
    # Format vocation into one-hot encoded columns
    for voc in VOCATIONS:
        if f'vocation_{voc}' not in character_data:
            character_data[f'vocation_{voc}'] = 0
    
    # Format as DataFrame
    if not isinstance(character_data, pd.DataFrame):
        character_data = pd.DataFrame([character_data])
    
    # Calculate any derived features
    if 'auction_duration_hours' not in character_data:
        character_data['auction_duration_hours'] = 48.0  # Default auction duration
    
    if 'level_to_bid_ratio' not in character_data:
        # We can't calculate this without the bid, which is what we're predicting
        # So use a typical ratio from the training data or set to 0
        character_data['level_to_bid_ratio'] = 0
    
    # Auction time features
    now = datetime.datetime.now()
    character_data['auction_hour'] = now.hour
    character_data['auction_day_of_week'] = now.weekday()
        
    # Preprocess the data
    X_new = preprocessor.transform(character_data)
    
    # Make predictions
    predictions = model.predict(X_new)
    
    return predictions[0]

# Function to handle Gradio interface prediction (simplified parameter list)
def predict_from_interface(level, vocation, server, axe, club, sword, distance, magic, 
                          mounts, outfits, gold, achievements, charms, special_char):
    # Get model and preprocessor
    model, preprocessor = load_model()
    if not model or not preprocessor:
        return 0, "Error: Could not load model or preprocessor"
    
    # Default values for simplified interface
    shielding = 80
    fishing = 10
    charm_points_available = 1000
    charm_points_spent = 500
    charm_expansion = False
    hunting_task_points = 100
    permanent_prey_slot = 0
    permanent_hunt_slot = 0
    prey_wildcards = 0
    hirelings = 0
    hirelings_jobs = 0
    hirelings_outfits = 0
    imbuements = 0
    
    # Prepare data in the format expected by the model
    character_data = {
        'level': level,
        'server': f"SomeServer {server}",  # Server name doesn't matter, just the suffix
        'is_name_contains_special_character': special_char,
        'axe_fighting': axe,
        'club_fighting': club,
        'sword_fighting': sword,
        'distance_fighting': distance,
        'fishing': fishing,
        'magic_level': magic,
        'shielding': shielding,
        'mounts': mounts,
        'outfits': outfits,
        'gold': gold,
        'achievement_points': achievements,
        'is_transfer_available': False,  # Default value
        'available_charm_points': charm_points_available,
        'spent_charm_points': charm_points_spent,
        'charm_expansion': charm_expansion,
        'hunting_task_points': hunting_task_points,
        'permanent_prey_task_slot': permanent_prey_slot,
        'permanent_hunt_task_slot': permanent_hunt_slot,
        'prey_wildcards': prey_wildcards,
        'hirelings': hirelings,
        'hirelings_jobs': hirelings_jobs,
        'hirelings_outfits': hirelings_outfits,
        'imbuements': imbuements,
        'charms': charms,
        'server_suffix': server
    }
    
    # Add vocation one-hot encoding
    for voc in VOCATIONS:
        character_data[f'vocation_{voc}'] = 1 if vocation == voc else 0
    
    # Add server suffix one-hot encoding
    for suffix in SERVER_SUFFIXES:
        if suffix != 'Optional-PvP':  # This is the reference category (drop_first=True)
            character_data[f'server_suffix_{suffix}'] = 1 if server == suffix else 0

    # Make prediction
    predicted_bid = predict_character_bid(character_data, model, preprocessor)
    
    # Format the result
    formatted_bid = f"{int(predicted_bid):,}"
    
    # Generate explanation
    explanation = f"""
    ## Character Bid Prediction
    
    Based on the provided attributes, this character is estimated to be worth approximately **{formatted_bid} gold coins**.
    
    ### Key factors affecting the price:
    - Character level: {level}
    - Vocation: {vocation.title()}
    - Server type: {server}
    - Magic level: {magic}
    - Combat skills: Avg. {(axe + club + sword + distance) / 4:.1f}
    
    *This prediction is based on historical auction data and market trends, and actual bids may vary.*
    """
    
    return predicted_bid, explanation

def launch_app():
    """Import Gradio and launch the app."""
    try:
        import gradio as gr
        
        # Create a simple interface
        iface = gr.Interface(
            fn=predict_from_interface,
            inputs=[
                # Main character info
                gr.Slider(minimum=8, maximum=2000, value=100, step=1, label="Character Level"),
                gr.Dropdown(choices=VOCATIONS, value="knight", label="Vocation"),
                gr.Dropdown(choices=SERVER_SUFFIXES, value="Optional-PvP", label="Server Type"),
                
                # Combat skills - simplified subset
                gr.Slider(minimum=10, maximum=130, value=80, step=1, label="Axe Fighting"),
                gr.Slider(minimum=10, maximum=130, value=10, step=1, label="Club Fighting"),
                gr.Slider(minimum=10, maximum=130, value=10, step=1, label="Sword Fighting"),
                gr.Slider(minimum=10, maximum=130, value=10, step=1, label="Distance Fighting"),
                gr.Slider(minimum=0, maximum=130, value=5, step=1, label="Magic Level"),
                
                # Character assets - simplified subset
                gr.Slider(minimum=0, maximum=100, value=5, step=1, label="Mounts"),
                gr.Slider(minimum=0, maximum=100, value=10, step=1, label="Outfits"),
                gr.Number(value=100000, label="Gold"),
                gr.Slider(minimum=0, maximum=2000, value=100, step=1, label="Achievement Points"),
                gr.Slider(minimum=0, maximum=20, value=0, step=1, label="Charms"),
                
                # Other
                gr.Checkbox(value=False, label="Name Contains Special Character"),
            ],
            outputs=[
                gr.Number(label="Predicted Bid (Gold Coins)"),
                gr.Markdown(label="Explanation"),
            ],
            title="Tibia Character Auction Price Predictor",
            description="Enter your character's attributes to predict the auction price",
            article="""<p>This tool predicts the auction price for Tibia characters based on an XGBoost machine learning model 
                      trained on historical auction data. Enter your character's attributes to get an estimated bid value.</p>
                      <p>The prediction is based on patterns found in past auctions and may not perfectly reflect current market conditions.</p>"""
        )
        
        # Launch the interface
        iface.launch(
            share=True,  # Create a public link
            server_name="0.0.0.0",  # Make it accessible from any IP
            server_port=7860  # Default Gradio port
        )
        
    except Exception as e:
        print(f"Error launching Gradio app: {e}")
        print("Please ensure Gradio is installed: pip install gradio")
        sys.exit(1)

# Run the app if the script is executed directly
if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please run the XGBoost notebook to train and save the model first.")
        sys.exit(1)
    
    # Launch the app
    launch_app()