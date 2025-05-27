#!/usr/bin/env python3
# Character Bid Prediction - Simple Gradio Web Interface
# This script creates a web interface for the trained XGBoost model
# that predicts character bid values based on their attributes.

import os
import sys
import pickle
import sqlite3
import pandas as pd
import numpy as np
import datetime
import json

# Add parent directory to path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Model path - use absolute path to avoid path issues
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'xgboost_character_bid_model.pkl')

# Load server information from servers.json
def load_server_info():
    """Load server information from servers.json file."""
    try:
        servers_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'servers.json')
        with open(servers_path, 'r') as f:
            servers_data = json.load(f)
        
        # Process servers data into a more usable format
        server_info = {}
        for server_name, server_data in servers_data.items():
            server_info[server_name] = {
                'server_location': server_data.get('serverLocation', {}).get('string', 'Unknown'),
                'pvp_type': server_data.get('pvpType', {}).get('string', 'Unknown'),
                'battleye': server_data.get('battleye', False),
                'server_experimental': server_data.get('experimental', False)
            }
        return server_info
    except Exception as e:
        print(f"Error loading server info: {e}")
        return {}

# Load server information
SERVER_INFO = load_server_info()

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

# Function to get actual server names from the database
def get_actual_servers():
    try:
        conn = sqlite3.connect(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'characters.db'))
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT server FROM characters ORDER BY server')
        servers = [row[0] for row in cursor.fetchall()]
        conn.close()
        return servers
    except Exception as e:
        print(f"Error retrieving server names: {e}")
        # Fallback to a predefined list if there's an error
        return [
            'Adra', 'Alumbra', 'Antica', 'Ardera', 'Astera', 'Axera', 'Bastia',
            'Batabra', 'Belobra', 'Bombra', 'Bona', 'Cadebra', 'Calmera', 'Castela',
            'Celebra', 'Celesta', 'Collabra', 'Damora', 'Descubra', 'Dibra'
        ]

# Get actual server names from the database
ACTUAL_SERVERS = get_actual_servers()

# These are the actual 3-letter suffixes the model expects (extracted from error message)
MODEL_SERVER_SUFFIXES = [
    'ira', 'sta', 'cia', 'una', 'ica', 'ona', 'ura', 'nia', 'ria', 'ela', 
    'oca', 'gia', 'osa', 'era', 'nza', 'rna', 'ora', 'dra', 'rsa', 'mia', 
    'ena', 'lsa', 'tia', 'ima'
]

# Function to make predictions
def predict_character_bid(character_data, model, preprocessor):
    """Make bid prediction for character data."""
    # Format as DataFrame
    if not isinstance(character_data, pd.DataFrame):
        character_data = pd.DataFrame([character_data])
    
    # Preprocess the data
    X_new = preprocessor.transform(character_data)
    
    # Make predictions
    predictions = model.predict(X_new)
    
    return predictions[0]

# Function to handle Gradio interface prediction (simplified parameter list)
def predict_from_interface(level, vocation, server, axe, club, sword, distance, magic, 
                          mounts, outfits, gold, transfer_available, achievements, charms, special_char):
    # Get model and preprocessor
    model, preprocessor = load_model()
    if not model or not preprocessor:
        return 0, "Error: Could not load model or preprocessor"
    
    # Default values for simplified interface
    shielding = 80
    fishing = 10
    fist_fighting = 10  # Add missing fist_fighting
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
    
    # Get server information from loaded data
    server_data = SERVER_INFO.get(server, {
        'server_location': 'Unknown',
        'pvp_type': 'Unknown', 
        'battleye': False,
        'server_experimental': False
    })
    
    # Prepare data in the format expected by the model
    character_data = {
        'level': level,
        'vocation': vocation,  # Keep as string, not one-hot encoded yet
        'server': server,
        'is_name_contains_special_character': special_char,
        'axe_fighting': axe,
        'club_fighting': club,
        'distance_fighting': distance,
        'fishing': fishing,
        'fist_fighting': fist_fighting,  # Add missing column
        'magic_level': magic,
        'shielding': shielding,
        'sword_fighting': sword,
        'mounts': mounts,
        'outfits': outfits,
        'gold': gold,
        'achievement_points': achievements,
        'is_transfer_available': transfer_available,
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
        # Add server metadata
        'server_location': server_data['server_location'],
        'pvp_type': server_data['pvp_type'],
        'battleye': server_data['battleye'],
        'server_experimental': server_data['server_experimental'],
        # Add auction time features
        'auction_duration_hours': 48.0,
        'auction_hour': datetime.datetime.now().hour,
        'auction_day_of_week': datetime.datetime.now().weekday()
    }
    
    # Make prediction
    predicted_bid = predict_character_bid(character_data, model, preprocessor)
    
    # Format the result
    formatted_bid = f"{int(predicted_bid):,}"
    
    # Generate explanation
    explanation = f"""
    ## Character Bid Prediction
    
    Based on the provided attributes, this character is estimated to be worth approximately **{formatted_bid} tibia coins (TC)**.
    
    ### Key factors affecting the price:
    - Character level: {level}
    - Vocation: {vocation.title()}
    - Server: {server}
    
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
                gr.Dropdown(choices=ACTUAL_SERVERS, value="Antica", label="Server"),
                
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
                gr.Checkbox(value=False, label="Transfer Available"),
                gr.Slider(minimum=0, maximum=2000, value=100, step=1, label="Achievement Points"),
                gr.Slider(minimum=0, maximum=20, value=0, step=1, label="Charms"),
                
                # Other
                gr.Checkbox(value=False, label="Name Contains Special Character"),
                
            ],
            outputs=[
                gr.Number(label="Predicted Bid (Tibia Coins)"),
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