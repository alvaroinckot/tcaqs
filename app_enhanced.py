#!/usr/bin/env python3
"""
Tibia Character Automated Quotation System (TCAQS) - Deployment Version
Fixed for Hugging Face Spaces compatibility with enhanced error handling
"""

import os
import sys
import pickle
import sqlite3
import pandas as pd
import numpy as np
import datetime
import json
import gradio as gr
import warnings

# Suppress scikit-learn version warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*sklearn.*")

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgboost_character_bid_model_v3.pkl')

def load_server_info():
    """Load server information from servers.json file."""
    try:
        servers_path = os.path.join(BASE_DIR, 'data', 'servers.json')
        with open(servers_path, 'r') as f:
            servers_data = json.load(f)
        
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

def load_model():
    """Load the XGBoost model and preprocessor with enhanced error handling."""
    try:
        if os.path.exists(MODEL_PATH):
            print(f"Attempting to load model from: {MODEL_PATH}")
            with open(MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
                return model_data['model'], model_data['preprocessor']
        else:
            print(f"Model not found at {MODEL_PATH}")
            return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        print("This might be due to scikit-learn version compatibility issues.")
        return None, None

def get_server_list():
    """Get list of available servers with robust fallback."""
    # Comprehensive fallback server list for deployment
    fallback_servers = [
        'Adra', 'Alumbra', 'Antica', 'Ardera', 'Astera', 'Axera', 'Bastia',
        'Batabra', 'Belobra', 'Bombra', 'Bona', 'Cadebra', 'Calmera', 'Castela',
        'Celebra', 'Celesta', 'Collabra', 'Damora', 'Descubra', 'Dibra',
        'Dolera', 'Emera', 'Esmera', 'Faluna', 'Ferobra', 'Fervora', 'Firmera',
        'Garnera', 'Gentebra', 'Gladera', 'Harmonia', 'Havera', 'Honbra',
        'Impulsa', 'Inabra', 'Javibra', 'Jupa', 'Kalibra', 'Kardera', 'Kenora',
        'Libertabra', 'Lobera', 'Luminera', 'Lutabra', 'Menera', 'Mirobra',
        'Monza', 'Nefera', 'Nossobra', 'Ombra', 'Peloria', 'Premia', 'Quelibra',
        'Refugia', 'Relania', 'Secura', 'Solidera', 'Talera', 'Tornera',
        'Venebra', 'Vunira', 'Wintera', 'Yonabra', 'Zuna', 'Zunera'
    ]
    
    try:
        db_path = os.path.join(BASE_DIR, 'data', 'characters.db')
        if not os.path.exists(db_path):
            print(f"Database not found at {db_path}, using fallback server list")
            return fallback_servers
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='characters'")
        if not cursor.fetchone():
            print("Characters table not found in database, using fallback server list")
            conn.close()
            return fallback_servers
            
        cursor.execute('SELECT DISTINCT server FROM characters ORDER BY server LIMIT 50')
        servers = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        if servers:
            print(f"Loaded {len(servers)} servers from database")
            return servers
        else:
            print("No servers found in database, using fallback list")
            return fallback_servers
            
    except Exception as e:
        print(f"Error getting servers from database: {e}, using fallback list")
        return fallback_servers

def create_simple_prediction(level, vocation, server, transfer_available, 
                           axe, club, sword, distance, magic, shielding, 
                           mounts, outfits, gold, achievements):
    """Create a simple rule-based prediction when ML model is unavailable."""
    
    # Simple heuristic based on common patterns in Tibia character auctions
    base_value = 0
    
    # Level contribution (most important factor)
    if level <= 50:
        base_value += level * 10
    elif level <= 100:
        base_value += 500 + (level - 50) * 25
    elif level <= 200:
        base_value += 1750 + (level - 100) * 50
    elif level <= 400:
        base_value += 6750 + (level - 200) * 100
    else:
        base_value += 26750 + (level - 400) * 150
    
    # Vocation multiplier
    vocation_multipliers = {
        'knight': 1.0,
        'paladin': 1.1,
        'sorcerer': 1.2,
        'druid': 1.15
    }
    base_value *= vocation_multipliers.get(vocation, 1.0)
    
    # Skills contribution
    primary_skill = max(axe, club, sword, distance)
    base_value += primary_skill * 25
    base_value += magic * 40
    base_value += shielding * 15
    
    # Assets contribution
    base_value += mounts * 100
    base_value += outfits * 50
    base_value += gold * 0.02  # 1 TC per 50 gold roughly
    base_value += achievements * 5
    
    # Transfer bonus
    if transfer_available:
        base_value *= 1.3
    
    # Server popularity bonus (simplified)
    popular_servers = ['Antica', 'Secura', 'Amera', 'Dolera']
    if server in popular_servers:
        base_value *= 1.1
    
    return max(int(base_value), 100)  # Minimum 100 TC

def predict_character_value(level, vocation, server, transfer_available, 
                          axe, club, sword, distance, magic, shielding, 
                          mounts, outfits, gold, achievements):
    """Predict character auction value with fallback to simple heuristics."""
    
    # Try to load and use ML model
    model, preprocessor = load_model()
    
    if model and preprocessor:
        try:
            # Prepare full feature set for ML model
            server_info = load_server_info()
            server_data = server_info.get(server, {
                'server_location': 'Unknown',
                'pvp_type': 'Unknown', 
                'battleye': False,
                'server_experimental': False
            })
            
            character_data = {
                'level': level,
                'axe_fighting': axe,
                'club_fighting': club,
                'sword_fighting': sword,
                'distance_fighting': distance,
                'magic_level': magic,
                'shielding': shielding,
                'mounts': mounts,
                'outfits': outfits,
                'gold': gold,
                'achievement_points': achievements,
                'is_transfer_available': transfer_available,
                'vocation': vocation,
                'server': server,
                'server_location': server_data['server_location'],
                'pvp_type': server_data['pvp_type'],
                'battleye': server_data['battleye'],
                'server_experimental': server_data['server_experimental'],
                # Required defaults
                'is_name_contains_special_character': False,
                'fishing': 10,
                'fist_fighting': 10,
                'available_charm_points': 1000,
                'spent_charm_points': 500,
                'charm_expansion': False,
                'hunting_task_points': 100,
                'permanent_prey_task_slot': 0,
                'permanent_hunt_task_slot': 0,
                'prey_wildcards': 0,
                'hirelings': 0,
                'hirelings_jobs': 0,
                'hirelings_outfits': 0,
                'imbuements': 0,
                'charms': 0,
                'auction_duration_hours': 48,
                'days_since_first_auction': 365,
                'days_since_server_first_auction': 200,
                'auction_start_date_iso': datetime.datetime.now().isoformat(),
                'auction_end_date_iso': (datetime.datetime.now() + datetime.timedelta(hours=48)).isoformat(),
            }
            
            # Add boolean features (simplified subset)
            boolean_features = [
                'imbuements_Powerful_Vampirism', 'imbuements_Powerful_Strike', 
                'charms_Divine_Wrath', 'quest_lines_The_Inquisition', 
                'mounts_War_Horse', 'store_mounts_Armoured_War_Horse'
            ]
            for feature in boolean_features:
                character_data[feature] = False
            
            # Convert to DataFrame and predict
            character_df = pd.DataFrame([character_data])
            X_new = preprocessor.transform(character_df)
            predicted_bid = model.predict(X_new)[0]
            
            explanation = f"""
## 🎮 Character Valuation Results (ML Model)

**Predicted Auction Value: {int(predicted_bid):,} Tibia Coins (TC)**

### 📊 Character Profile
- **Level {level} {vocation.title()}** on {server}
- **Transfer Available**: {'✅ Yes' if transfer_available else '❌ No'}

### ⚔️ Combat Skills
- **Melee**: Axe {axe} | Club {club} | Sword {sword}
- **Distance**: {distance} | **Magic Level**: {magic} | **Shielding**: {shielding}

### 💎 Character Assets  
- **Mounts**: {mounts} | **Outfits**: {outfits}
- **Gold**: {gold:,} | **Achievement Points**: {achievements}

---
*Prediction based on XGBoost model trained on 650k+ character auctions from 2022*
            """
            
            return int(predicted_bid), explanation
            
        except Exception as e:
            print(f"ML prediction failed: {e}, falling back to heuristic method")
    
    # Fallback to simple heuristic prediction
    predicted_bid = create_simple_prediction(
        level, vocation, server, transfer_available, 
        axe, club, sword, distance, magic, shielding, 
        mounts, outfits, gold, achievements
    )
    
    explanation = f"""
## 🎮 Character Valuation Results (Heuristic Model)

**Estimated Auction Value: {predicted_bid:,} Tibia Coins (TC)**

### 📊 Character Profile
- **Level {level} {vocation.title()}** on {server}
- **Transfer Available**: {'✅ Yes' if transfer_available else '❌ No'}

### ⚔️ Combat Skills
- **Melee**: Axe {axe} | Club {club} | Sword {sword}
- **Distance**: {distance} | **Magic Level**: {magic} | **Shielding**: {shielding}

### 💎 Character Assets  
- **Mounts**: {mounts} | **Outfits**: {outfits}
- **Gold**: {gold:,} | **Achievement Points**: {achievements}

---
⚠️ *This prediction uses simplified heuristics due to ML model unavailability. For more accurate predictions, the full ML model would need compatible dependencies.*

**Note**: Estimates are based on general Tibia market patterns and may not reflect current conditions.
    """
    
    return predicted_bid, explanation

# Initialize data
print("Loading TCAQS...")
SERVERS = get_server_list()
VOCATIONS = ['knight', 'paladin', 'sorcerer', 'druid']

# Create Gradio interface
def create_interface():
    with gr.Blocks(
        title="TCAQS - Tibia Character Price Predictor",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown("""
# 🎮 Tibia Character Automated Quotation System (TCAQS)

Predict your character's auction value using machine learning trained on 650,000+ historical auctions.

**🔗 Source**: [GitHub Repository](https://github.com/alvaroinckot/tcaqs)
        """)
        
        with gr.Tab("🏆 Basic Information"):
            with gr.Row():
                with gr.Column():
                    level = gr.Slider(8, 2000, 100, step=1, label="Character Level")
                    vocation = gr.Dropdown(VOCATIONS, value="knight", label="Vocation")
                    server = gr.Dropdown(SERVERS, value=SERVERS[0] if SERVERS else "Antica", label="Server")
                    transfer_available = gr.Checkbox(False, label="Transfer Available")
        
        with gr.Tab("⚔️ Combat Skills"):
            with gr.Row():
                with gr.Column():
                    axe = gr.Slider(10, 140, 80, step=1, label="Axe Fighting")
                    club = gr.Slider(10, 140, 10, step=1, label="Club Fighting")
                    sword = gr.Slider(10, 140, 10, step=1, label="Sword Fighting")
                with gr.Column():
                    distance = gr.Slider(10, 140, 10, step=1, label="Distance Fighting")
                    magic = gr.Slider(0, 130, 5, step=1, label="Magic Level")
                    shielding = gr.Slider(10, 140, 80, step=1, label="Shielding")
        
        with gr.Tab("💎 Character Assets"):
            with gr.Row():
                with gr.Column():
                    mounts = gr.Slider(0, 100, 5, step=1, label="Mounts")
                    outfits = gr.Slider(0, 100, 10, step=1, label="Outfits")
                with gr.Column():
                    gold = gr.Number(100000, label="Gold")
                    achievements = gr.Slider(0, 2000, 100, step=1, label="Achievement Points")
        
        # Prediction
        with gr.Row():
            predict_btn = gr.Button("🔮 Predict Character Value", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                predicted_value = gr.Number(label="Predicted Value (TC)", precision=0)
            with gr.Column():
                explanation = gr.Markdown(label="Detailed Analysis")
        
        # Connect prediction
        predict_btn.click(
            fn=predict_character_value,
            inputs=[level, vocation, server, transfer_available, 
                   axe, club, sword, distance, magic, shielding, 
                   mounts, outfits, gold, achievements],
            outputs=[predicted_value, explanation]
        )
        
        # Examples
        gr.Examples(
            examples=[
                [100, "knight", SERVERS[0] if SERVERS else "Antica", False, 80, 10, 10, 10, 5, 80, 5, 10, 100000, 100],
                [500, "paladin", SERVERS[1] if len(SERVERS) > 1 else "Secura", True, 15, 15, 15, 120, 30, 100, 20, 30, 5000000, 800],
                [250, "sorcerer", SERVERS[2] if len(SERVERS) > 2 else "Amera", False, 10, 10, 10, 10, 80, 20, 15, 25, 1000000, 400]
            ],
            inputs=[level, vocation, server, transfer_available, 
                   axe, club, sword, distance, magic, shielding, 
                   mounts, outfits, gold, achievements],
            label="Example Characters"
        )
        
        gr.Markdown("""
### 📊 Model Information

- **Algorithm**: XGBoost Regression (with heuristic fallback)
- **Accuracy**: R² = 0.9349 (93.49%) when ML model available
- **Training Data**: 650,000+ character auctions from 2022
- **Features**: 400+ engineered features including skills, assets, and market factors

**Current Status**: 
- ✅ Interface functional
- ⚠️ ML model availability depends on environment compatibility
- ✅ Heuristic fallback provides reasonable estimates

*Note: Predictions are estimates based on 2022 historical data and may not reflect current market conditions.*

**Acknowledgments**: Data sourced from the [Exevo Pan](https://exevopan.com) project.
        """)
    
    return demo

# Create the app
demo = create_interface()

# Make available for deployment
app = demo
iface = demo

print("TCAQS interface created successfully!")

if __name__ == "__main__":
    print("Starting TCAQS application...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
