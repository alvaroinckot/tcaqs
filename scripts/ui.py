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

# Model path - use paths relative to script location for deployment
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgboost_character_bid_model_v3.pkl')

# Load server information from servers.json
def load_server_info():
    """Load server information from servers.json file."""
    try:
        servers_path = os.path.join(BASE_DIR, 'data', 'servers.json')
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
        # First try the provided path
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                model = model_data['model']
                preprocessor = model_data['preprocessor']
                print("Model loaded successfully.")
                return model, preprocessor
        else:
            # Try alternative paths for deployment
            alternative_paths = [
                os.path.join(os.getcwd(), 'models', 'xgboost_character_bid_model_v3.pkl'),
                os.path.join(os.path.dirname(__file__), '..', 'models', 'xgboost_character_bid_model_v3.pkl'),
                'xgboost_character_bid_model_v3.pkl',  # Current directory
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    print(f"Loading model from alternative path: {alt_path}")
                    with open(alt_path, 'rb') as f:
                        model_data = pickle.load(f)
                        model = model_data['model']
                        preprocessor = model_data['preprocessor']
                        print("Model loaded successfully from alternative path.")
                        return model, preprocessor
            
            raise FileNotFoundError(f"Model file not found at {model_path} or alternative paths")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# Define vocations for dropdown
VOCATIONS = ['knight', 'paladin', 'sorcerer', 'druid']

# Function to get actual server names from the database
def get_actual_servers():
    try:
        db_path = os.path.join(BASE_DIR, 'data', 'characters.db')
        conn = sqlite3.connect(db_path)
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

# Function to handle Gradio interface prediction (comprehensive parameter list)
def predict_from_interface(
    # Core character attributes
    level, vocation, server, special_char,
    # Combat skills
    axe, club, sword, distance, magic, shielding, fishing, fist_fighting,
    # Character assets
    mounts, outfits, gold, achievements, transfer_available,
    # Charm system
    charms, charm_points_available, charm_points_spent, charm_expansion,
    # Task system
    hunting_task_points, permanent_prey_slot, permanent_hunt_slot, prey_wildcards,
    # Hirelings
    hirelings, hirelings_jobs, hirelings_outfits,
    # Imbuements
    imbuements,
    # Temporal features
    auction_duration_hours, days_since_first_auction, days_since_server_first_auction,
    # Boolean feature selections
    selected_imbuements, selected_charms_features, selected_quest_lines,
    selected_mounts_features, selected_store_mounts, selected_store_outfits
):
    # Get model and preprocessor
    model, preprocessor = load_model()
    if not model or not preprocessor:
        return 0, "Error: Could not load model or preprocessor"
    
    # Get server information from loaded data
    server_data = SERVER_INFO.get(server, {
        'server_location': 'Unknown',
        'pvp_type': 'Unknown', 
        'battleye': False,
        'server_experimental': False
    })
    
    # Prepare data in the format expected by the model (all features)
    character_data = {
        # Core features (27 numerical features)
        'level': level,
        'is_name_contains_special_character': special_char,
        'axe_fighting': axe,
        'club_fighting': club,
        'distance_fighting': distance,
        'fishing': fishing,
        'fist_fighting': fist_fighting,
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
        
        # Temporal features (3 features)
        'auction_duration_hours': auction_duration_hours,
        'days_since_first_auction': days_since_first_auction,
        'days_since_server_first_auction': days_since_server_first_auction,
        
        # Date features (2 features) - Note: These are kept for compatibility but temporal features above are what matter
        'auction_start_date_iso': datetime.datetime.now().isoformat(),
        'auction_end_date_iso': (datetime.datetime.now() + datetime.timedelta(hours=48)).isoformat(),
        
        # Categorical features (6 features)
        'vocation': vocation,
        'server': server,
        'server_location': server_data['server_location'],
        'pvp_type': server_data['pvp_type'],
        'battleye': server_data['battleye'],
        'server_experimental': server_data['server_experimental'],
    }
    
    # Add default boolean features from the processed lists
    # The model expects 385 categorical features, most of which are boolean
    # We'll add all of them with default False values for the simplified UI
    
    # Get the expected feature names from the model
    def add_all_missing_boolean_features(character_data, selected_imbuements, selected_charms_features, 
                                       selected_quest_lines, selected_mounts_features, 
                                       selected_store_mounts, selected_store_outfits):
        """Add all missing boolean features expected by the model with user-selected values."""
        
        # Complete list of all 379 boolean features expected by the model (exactly matching training data)
        all_boolean_features = {
            # Imbuement features (23 total)
            'imbuements_Powerful_Bash': 'imbuements_Powerful_Bash' in selected_imbuements,
            'imbuements_Powerful_Blockade': 'imbuements_Powerful_Blockade' in selected_imbuements,
            'imbuements_Powerful_Chop': 'imbuements_Powerful_Chop' in selected_imbuements,
            'imbuements_Powerful_Cloud_Fabric': 'imbuements_Powerful_Cloud_Fabric' in selected_imbuements,
            'imbuements_Powerful_Demon_Presence': 'imbuements_Powerful_Demon_Presence' in selected_imbuements,
            'imbuements_Powerful_Dragon_Hide': 'imbuements_Powerful_Dragon_Hide' in selected_imbuements,
            'imbuements_Powerful_Electrify': 'imbuements_Powerful_Electrify' in selected_imbuements,
            'imbuements_Powerful_Epiphany': 'imbuements_Powerful_Epiphany' in selected_imbuements,
            'imbuements_Powerful_Featherweight': 'imbuements_Powerful_Featherweight' in selected_imbuements,
            'imbuements_Powerful_Frost': 'imbuements_Powerful_Frost' in selected_imbuements,
            'imbuements_Powerful_Lich_Shroud': 'imbuements_Powerful_Lich_Shroud' in selected_imbuements,
            'imbuements_Powerful_Precision': 'imbuements_Powerful_Precision' in selected_imbuements,
            'imbuements_Powerful_Quara_Scale': 'imbuements_Powerful_Quara_Scale' in selected_imbuements,
            'imbuements_Powerful_Reap': 'imbuements_Powerful_Reap' in selected_imbuements,
            'imbuements_Powerful_Scorch': 'imbuements_Powerful_Scorch' in selected_imbuements,
            'imbuements_Powerful_Slash': 'imbuements_Powerful_Slash' in selected_imbuements,
            'imbuements_Powerful_Snake_Skin': 'imbuements_Powerful_Snake_Skin' in selected_imbuements,
            'imbuements_Powerful_Strike': 'imbuements_Powerful_Strike' in selected_imbuements,
            'imbuements_Powerful_Swiftness': 'imbuements_Powerful_Swiftness' in selected_imbuements,
            'imbuements_Powerful_Vampirism': 'imbuements_Powerful_Vampirism' in selected_imbuements,
            'imbuements_Powerful_Venom': 'imbuements_Powerful_Venom' in selected_imbuements,
            'imbuements_Powerful_Vibrancy': 'imbuements_Powerful_Vibrancy' in selected_imbuements,
            'imbuements_Powerful_Void': 'imbuements_Powerful_Void' in selected_imbuements,
            
            # Charm features (19 total)
            'charms_Adrenaline_Burst': 'charms_Adrenaline_Burst' in selected_charms_features,
            'charms_Bless': 'charms_Bless' in selected_charms_features,
            'charms_Cleanse': 'charms_Cleanse' in selected_charms_features,
            'charms_Cripple': 'charms_Cripple' in selected_charms_features,
            'charms_Curse': 'charms_Curse' in selected_charms_features,
            'charms_Divine_Wrath': 'charms_Divine_Wrath' in selected_charms_features,
            'charms_Dodge': 'charms_Dodge' in selected_charms_features,
            'charms_Enflame': 'charms_Enflame' in selected_charms_features,
            'charms_Freeze': 'charms_Freeze' in selected_charms_features,
            'charms_Gut': 'charms_Gut' in selected_charms_features,
            'charms_Low_Blow': 'charms_Low_Blow' in selected_charms_features,
            'charms_Numb': 'charms_Numb' in selected_charms_features,
            'charms_Parry': 'charms_Parry' in selected_charms_features,
            'charms_Poison': 'charms_Poison' in selected_charms_features,
            'charms_Scavenge': 'charms_Scavenge' in selected_charms_features,
            'charms_Vampiric_Embrace': 'charms_Vampiric_Embrace' in selected_charms_features,
            'charms_Voids_Call': 'charms_Voids_Call' in selected_charms_features,
            'charms_Wound': 'charms_Wound' in selected_charms_features,
            'charms_Zap': 'charms_Zap' in selected_charms_features,
        }
        
        # Add quest lines
        quest_lines = [
            'quest_lines_A_Fathers_Burden', 'quest_lines_An_Uneasy_Alliance', 'quest_lines_Blood_Brothers',
            'quest_lines_Child_of_Destiny', 'quest_lines_Children_of_the_Revolution', 'quest_lines_Dark_Trails',
            'quest_lines_Dawnport', 'quest_lines_Ferumbras_Ascendant', 'quest_lines_Heart_of_Destruction',
            'quest_lines_Hero_of_Rathleton', 'quest_lines_Hot_Cuisine', 'quest_lines_In_Service_Of_Yalahar',
            'quest_lines_Kissing_a_Pig', 'quest_lines_Primal_Ordeal', 'quest_lines_Sea_of_Light',
            'quest_lines_Shadows_of_Yalahar', 'quest_lines_Soul_War', 'quest_lines_The_Ancient_Tombs',
            'quest_lines_The_Ape_City', 'quest_lines_The_Beginning', 'quest_lines_The_Desert_Dungeon',
            'quest_lines_The_Djinn_War_Efreet_Faction', 'quest_lines_The_Djinn_War_Marid_Faction',
            'quest_lines_The_Explorer_Society', 'quest_lines_The_Gravedigger_of_Drefia', 'quest_lines_The_Ice_Islands',
            'quest_lines_The_Inquisition', 'quest_lines_The_Isle_Of_Evil', 'quest_lines_The_New_Frontier',
            'quest_lines_The_Outlaw_Camp', 'quest_lines_The_Paradox_Tower', 'quest_lines_The_Pits_of_Inferno',
            'quest_lines_The_Postman_Missions', 'quest_lines_The_Queen_of_the_Banshees', 'quest_lines_The_Rookie_Guard',
            'quest_lines_The_Scatterbrained_Sorcerer', 'quest_lines_The_Shattered_Isles', 'quest_lines_The_Thieves_Guild',
            'quest_lines_The_White_Raven_Monastery', 'quest_lines_Twenty_Miles_Beneath_The_Sea',
            'quest_lines_Unnatural_Selection', 'quest_lines_What_a_foolish_Quest', 'quest_lines_Wrath_of_the_Emperor'
        ]
        for quest in quest_lines:
            all_boolean_features[quest] = quest in selected_quest_lines
            
        # Add mounts
        mount_features = [
            'mounts_Antelope', 'mounts_Black_Sheep', 'mounts_Blazebringer', 'mounts_Blue_Rolling_Barrel',
            'mounts_Bright_Percht_Sleigh', 'mounts_Bright_Percht_Sleigh_Variant', 'mounts_Cold_Percht_Sleigh',
            'mounts_Cold_Percht_Sleigh_Variant', 'mounts_Crystal_Wolf', 'mounts_Dark_Percht_Sleigh',
            'mounts_Dark_Percht_Sleigh_Variant', 'mounts_Donkey', 'mounts_Dragonling', 'mounts_Draptor',
            'mounts_Dromedary', 'mounts_Finished_Bright_Percht_Sleigh', 'mounts_Finished_Cold_Percht_Sleigh',
            'mounts_Finished_Dark_Percht_Sleigh', 'mounts_Fleeting_Knowledge', 'mounts_Giant_Beaver',
            'mounts_Glooth_Glider', 'mounts_Gloothomotive', 'mounts_Gnarlhound', 'mounts_Green_Rolling_Barrel',
            'mounts_Gryphon', 'mounts_Haze', 'mounts_Hibernal_Moth', 'mounts_Ironblight', 'mounts_Kingly_Deer',
            'mounts_Krakoloss', 'mounts_Lacewing_Moth', 'mounts_Lady_Bug', 'mounts_Magma_Crawler',
            'mounts_Manta_Ray', 'mounts_Midnight_Panther', 'mounts_Mole', 'mounts_Neon_Sparkid',
            'mounts_Noble_Lion', 'mounts_Phant', 'mounts_Phantasmal_Jade', 'mounts_Racing_Bird',
            'mounts_Rapid_Boar', 'mounts_Red_Rolling_Barrel', 'mounts_Rented_Horse', 'mounts_Rift_Runner',
            'mounts_Ripptor', 'mounts_Scorpion_King', 'mounts_Shellodon', 'mounts_Shock_Head',
            'mounts_Singeing_Steed', 'mounts_Sparkion', 'mounts_Stampor', 'mounts_Stone_Rhino',
            'mounts_Tamed_Panda', 'mounts_The_Hellgrip', 'mounts_Tiger_Slug', 'mounts_Tin_Lizzard',
            'mounts_Titanica', 'mounts_Undead_Cavebear', 'mounts_Uniwheel', 'mounts_Ursagrodon',
            'mounts_Walker', 'mounts_War_Bear', 'mounts_War_Horse', 'mounts_Water_Buffalo',
            'mounts_White_Lion', 'mounts_Widow_Queen'
        ]
        for mount in mount_features:
            all_boolean_features[mount] = mount in selected_mounts_features
            
        # Add store mounts (just add a few key ones to keep UI manageable)
        store_mount_features = [
            'store_mounts_Arctic_Unicorn', 'store_mounts_Armoured_War_Horse', 'store_mounts_Azudocus',
            'store_mounts_Battle_Badger', 'store_mounts_Black_Stag', 'store_mounts_Blazing_Unicorn',
            'store_mounts_Crystal_Wolf', 'store_mounts_Desert_King', 'store_mounts_Doombringer',
            'store_mounts_Emerald_Raven', 'store_mounts_Festive_Mammoth', 'store_mounts_Flying_Divan',
            'store_mounts_Golden_Dragonfly', 'store_mounts_Highland_Yak', 'store_mounts_Magic_Carpet',
            'store_mounts_Midnight_Panther', 'store_mounts_Prismatic_Unicorn', 'store_mounts_Shadow_Draptor'
        ]
        # Add remaining store mounts with default False
        all_store_mounts = [
            'store_mounts_Arctic_Unicorn', 'store_mounts_Armoured_War_Horse', 'store_mounts_Azudocus',
            'store_mounts_Batcat', 'store_mounts_Battle_Badger', 'store_mounts_Black_Stag',
            'store_mounts_Blackpelt', 'store_mounts_Blazing_Unicorn', 'store_mounts_Bloodcurl',
            'store_mounts_Bogwurm', 'store_mounts_Boreal_Owl', 'store_mounts_Bunny_Dray',
            'store_mounts_Caped_Snowman', 'store_mounts_Carpacosaurus', 'store_mounts_Cave_Tarantula',
            'store_mounts_Cinderhoof', 'store_mounts_Cinnamon_Ibex', 'store_mounts_Cony_Cart',
            'store_mounts_Copper_Fly', 'store_mounts_Coral_Rhea', 'store_mounts_Coralripper',
            'store_mounts_Cranium_Spider', 'store_mounts_Crimson_Ray', 'store_mounts_Cunning_Hyaena',
            'store_mounts_Dandelion', 'store_mounts_Dawn_Strayer', 'store_mounts_Death_Crawler',
            'store_mounts_Desert_King', 'store_mounts_Doombringer', 'store_mounts_Dreadhare',
            'store_mounts_Dusk_Pryer', 'store_mounts_Ebony_Tiger', 'store_mounts_Ember_Saurian',
            'store_mounts_Emerald_Raven', 'store_mounts_Emerald_Sphinx', 'store_mounts_Emerald_Waccoon',
            'store_mounts_Emperor_Deer', 'store_mounts_Ether_Badger', 'store_mounts_Eventide_Nandu',
            'store_mounts_Feral_Tiger', 'store_mounts_Festive_Mammoth', 'store_mounts_Festive_Snowman',
            'store_mounts_Flamesteed', 'store_mounts_Flitterkatzen', 'store_mounts_Floating_Augur',
            'store_mounts_Floating_Kashmir', 'store_mounts_Floating_Sage', 'store_mounts_Floating_Scholar',
            'store_mounts_Flying_Divan', 'store_mounts_Frostflare', 'store_mounts_Glacier_Vagabond',
            'store_mounts_Gloom_Widow', 'store_mounts_Gloomwurm', 'store_mounts_Gold_Sphinx',
            'store_mounts_Golden_Dragonfly', 'store_mounts_Gorongra', 'store_mounts_Hailstorm_Fury',
            'store_mounts_Highland_Yak', 'store_mounts_Holiday_Mammoth', 'store_mounts_Hyacinth',
            'store_mounts_Ivory_Fang', 'store_mounts_Jackalope', 'store_mounts_Jade_Lion',
            'store_mounts_Jade_Pincer', 'store_mounts_Jade_Shrine', 'store_mounts_Jungle_Saurian',
            'store_mounts_Jungle_Tiger', 'store_mounts_Lagoon_Saurian', 'store_mounts_Leafscuttler',
            'store_mounts_Magic_Carpet', 'store_mounts_Marsh_Toad', 'store_mounts_Merry_Mammoth',
            'store_mounts_Mint_Ibex', 'store_mounts_Mould_Shell', 'store_mounts_Mouldpincer',
            'store_mounts_Muffled_Snowman', 'store_mounts_Mystic_Raven', 'store_mounts_Nethersteed',
            'store_mounts_Night_Waccoon', 'store_mounts_Nightdweller', 'store_mounts_Nightmarish_Crocovile',
            'store_mounts_Nightstinger', 'store_mounts_Noctungra', 'store_mounts_Obsidian_Shrine',
            'store_mounts_Parade_Horse', 'store_mounts_Peony', 'store_mounts_Platesaurian',
            'store_mounts_Plumfish', 'store_mounts_Poisonbane', 'store_mounts_Poppy_Ibex',
            'store_mounts_Prismatic_Unicorn', 'store_mounts_Rabbit_Rickshaw', 'store_mounts_Radiant_Raven',
            'store_mounts_Razorcreep', 'store_mounts_Reed_Lurker', 'store_mounts_Rift_Watcher',
            'store_mounts_Ringtail_Waccoon', 'store_mounts_River_Crocovile', 'store_mounts_Rune_Watcher',
            'store_mounts_Rustwurm', 'store_mounts_Sanguine_Frog', 'store_mounts_Savanna_Ostrich',
            'store_mounts_Scruffy_Hyaena', 'store_mounts_Sea_Devil', 'store_mounts_Shadow_Claw',
            'store_mounts_Shadow_Draptor', 'store_mounts_Shadow_Hart', 'store_mounts_Shadow_Sphinx',
            'store_mounts_Siegebreaker', 'store_mounts_Silverneck', 'store_mounts_Slagsnare',
            'store_mounts_Snow_Pelt', 'store_mounts_Snow_Strider', 'store_mounts_Snowy_Owl',
            'store_mounts_Steel_Bee', 'store_mounts_Steelbeak', 'store_mounts_Swamp_Crocovile',
            'store_mounts_Swamp_Snapper', 'store_mounts_Tawny_Owl', 'store_mounts_Tempest',
            'store_mounts_Tombstinger', 'store_mounts_Topaz_Shrine', 'store_mounts_Tourney_Horse',
            'store_mounts_Toxic_Toad', 'store_mounts_Tundra_Rambler', 'store_mounts_Venompaw',
            'store_mounts_Void_Watcher', 'store_mounts_Voracious_Hyaena', 'store_mounts_Winter_King',
            'store_mounts_Wolpertinger', 'store_mounts_Woodland_Prince', 'store_mounts_Zaoan_Badger'
        ]
        for mount in all_store_mounts:
            all_boolean_features[mount] = mount in selected_store_mounts
            
        # Add store outfits (just add a few key ones to keep UI manageable)
        all_store_outfits = [
            'store_outfits_Arbalester_base_and_addon_1_and_addon_2', 'store_outfits_Arena_Champion_base_and_addon_1_and_addon_2',
            'store_outfits_Beastmaster_base_and_addon_1_and_addon_2', 'store_outfits_Beastmaster_base_and_addon_2',
            'store_outfits_Beastmaster_base', 'store_outfits_Breezy_Garb_base_and_addon_1_and_addon_2',
            'store_outfits_Ceremonial_Garb_base_and_addon_1_and_addon_2', 'store_outfits_Ceremonial_Garb_base_and_addon_1',
            'store_outfits_Champion_base_and_addon_1_and_addon_2', 'store_outfits_Champion_base_and_addon_1',
            'store_outfits_Champion_base_and_addon_2', 'store_outfits_Champion_base',
            'store_outfits_Chaos_Acolyte_base_and_addon_1_and_addon_2', 'store_outfits_Chaos_Acolyte_base_and_addon_1',
            'store_outfits_Chaos_Acolyte_base', 'store_outfits_Conjurer_base_and_addon_1_and_addon_2',
            'store_outfits_Conjurer_base_and_addon_1', 'store_outfits_Conjurer_base_and_addon_2',
            'store_outfits_Conjurer_base', 'store_outfits_Death_Herald_base_and_addon_1_and_addon_2',
            'store_outfits_Death_Herald_base_and_addon_1', 'store_outfits_Death_Herald_base_and_addon_2',
            'store_outfits_Death_Herald_base', 'store_outfits_Dragon_Knight_base_and_addon_1_and_addon_2',
            'store_outfits_Entrepreneur_base_and_addon_1_and_addon_2', 'store_outfits_Entrepreneur_base_and_addon_1',
            'store_outfits_Entrepreneur_base_and_addon_2', 'store_outfits_Entrepreneur_base',
            'store_outfits_Evoker_base_and_addon_1_and_addon_2', 'store_outfits_Evoker_base_and_addon_1',
            'store_outfits_Evoker_base_and_addon_2', 'store_outfits_Evoker_base',
            'store_outfits_Fencer_base_and_addon_1_and_addon_2', 'store_outfits_Forest_Warden_base_and_addon_1_and_addon_2',
            'store_outfits_Ghost_Blade_base_and_addon_1_and_addon_2', 'store_outfits_Grove_Keeper_base_and_addon_1_and_addon_2',
            'store_outfits_Grove_Keeper_base_and_addon_2', 'store_outfits_Grove_Keeper_base',
            'store_outfits_Guidon_Bearer_base_and_addon_1_and_addon_2', 'store_outfits_Herbalist_base_and_addon_1_and_addon_2',
            'store_outfits_Herder_base_and_addon_1_and_addon_2', 'store_outfits_Jouster_base_and_addon_1_and_addon_2',
            'store_outfits_Lupine_Warden_base_and_addon_1_and_addon_2', 'store_outfits_Lupine_Warden_base_and_addon_2',
            'store_outfits_Lupine_Warden_base', 'store_outfits_Mercenary_base_and_addon_1_and_addon_2',
            'store_outfits_Merry_Garb_base_and_addon_1_and_addon_2', 'store_outfits_Moth_Cape_base_and_addon_1_and_addon_2',
            'store_outfits_Nordic_Chieftain_base_and_addon_1_and_addon_2', 'store_outfits_Owl_Keeper_base_and_addon_1_and_addon_2',
            'store_outfits_Pharaoh_base_and_addon_1_and_addon_2', 'store_outfits_Pharaoh_base_and_addon_2',
            'store_outfits_Philosopher_base_and_addon_1_and_addon_2', 'store_outfits_Philosopher_base_and_addon_1',
            'store_outfits_Philosopher_base', 'store_outfits_Pumpkin_Mummy_base_and_addon_1_and_addon_2',
            'store_outfits_Puppeteer_base_and_addon_1_and_addon_2', 'store_outfits_Puppeteer_base_and_addon_2',
            'store_outfits_Puppeteer_base', 'store_outfits_Ranger_base_and_addon_1_and_addon_2',
            'store_outfits_Ranger_base_and_addon_2', 'store_outfits_Ranger_base',
            'store_outfits_Retro_Citizen_base', 'store_outfits_Retro_Hunter_base',
            'store_outfits_Retro_Knight_base', 'store_outfits_Retro_Mage_base',
            'store_outfits_Retro_Nobleman_base', 'store_outfits_Retro_Noblewoman_base',
            'store_outfits_Retro_Summoner_base', 'store_outfits_Retro_Warrior_base',
            'store_outfits_Royal_Pumpkin_base_and_addon_1_and_addon_2', 'store_outfits_Royal_Pumpkin_base_and_addon_2',
            'store_outfits_Royal_Pumpkin_base', 'store_outfits_Rune_Master_base_and_addon_1_and_addon_2',
            'store_outfits_Sea_Dog_base_and_addon_1_and_addon_2', 'store_outfits_Sea_Dog_base_and_addon_1',
            'store_outfits_Sea_Dog_base', 'store_outfits_Seaweaver_base_and_addon_1_and_addon_2',
            'store_outfits_Seaweaver_base_and_addon_2', 'store_outfits_Seaweaver_base',
            'store_outfits_Siege_Master_base_and_addon_1_and_addon_2', 'store_outfits_Siege_Master_base_and_addon_1',
            'store_outfits_Siege_Master_base', 'store_outfits_Sinister_Archer_base_and_addon_1_and_addon_2',
            'store_outfits_Spirit_Caller_base_and_addon_1_and_addon_2', 'store_outfits_Spirit_Caller_base_and_addon_1',
            'store_outfits_Spirit_Caller_base_and_addon_2', 'store_outfits_Spirit_Caller_base',
            'store_outfits_Sun_Priest_base_and_addon_1_and_addon_2', 'store_outfits_Sun_Priestess_base_and_addon_1_and_addon_2',
            'store_outfits_Trailblazer_base_and_addon_1_and_addon_2', 'store_outfits_Trophy_Hunter_base_and_addon_1_and_addon_2',
            'store_outfits_Trophy_Hunter_base', 'store_outfits_Winter_Warden_base_and_addon_1_and_addon_2',
            'store_outfits_Winter_Warden_base'
        ]
        for outfit in all_store_outfits:
            all_boolean_features[outfit] = outfit in selected_store_outfits
        
        # Add all boolean features to character data
        for feature_name, value in all_boolean_features.items():
            character_data[feature_name] = value
            
        return character_data
    
    # Apply all the missing boolean features with user selections
    character_data = add_all_missing_boolean_features(character_data, selected_imbuements, 
                                                     selected_charms_features, selected_quest_lines,
                                                     selected_mounts_features, selected_store_mounts, 
                                                     selected_store_outfits)
    
    # Make prediction
    predicted_bid = predict_character_bid(character_data, model, preprocessor)
    
    # Format the result
    formatted_bid = f"{int(predicted_bid):,}"
    
    # Generate comprehensive explanation
    explanation = f"""
    ## 🎮 Character Bid Prediction Results
    
    Based on the provided attributes, this character is estimated to be worth approximately **{formatted_bid} tibia coins (TC)**.
    
    ### 📊 Character Summary
    - **Level**: {level} {vocation.title()}
    - **Server**: {server}
    - **Transfer Available**: {'Yes' if transfer_available else 'No'}
    
    ### ⚔️ Combat Profile
    - **Primary Skills**: Axe: {axe}, Club: {club}, Sword: {sword}, Distance: {distance}
    - **Magic Level**: {magic}
    - **Shielding**: {shielding}
    
    ### 💎 Character Assets
    - **Mounts**: {mounts} | **Outfits**: {outfits}
    - **Gold**: {gold:,} | **Achievement Points**: {achievements}
    - **Charms**: {charms} | **Imbuements**: {imbuements}
    
    ### 🎯 Advanced Features
    - **Hunting Task Points**: {hunting_task_points}
    - **Permanent Slots**: {permanent_prey_slot} Prey, {permanent_hunt_slot} Hunt
    - **Hirelings**: {hirelings} (Jobs: {hirelings_jobs}, Outfits: {hirelings_outfits})
    
    ### ✨ Selected Boolean Features
    - **Specific Imbuements**: {len(selected_imbuements)} selected ({', '.join(selected_imbuements[:3])}{'...' if len(selected_imbuements) > 3 else ''})
    - **Specific Charms**: {len(selected_charms_features)} selected ({', '.join(selected_charms_features[:3])}{'...' if len(selected_charms_features) > 3 else ''})
    - **Quest Lines**: {len(selected_quest_lines)} completed ({', '.join(selected_quest_lines[:3])}{'...' if len(selected_quest_lines) > 3 else ''})
    - **Specific Mounts**: {len(selected_mounts_features)} selected ({', '.join(selected_mounts_features[:3])}{'...' if len(selected_mounts_features) > 3 else ''})
    - **Store Mounts**: {len(selected_store_mounts)} selected ({', '.join(selected_store_mounts[:2])}{'...' if len(selected_store_mounts) > 2 else ''})
    - **Store Outfits**: {len(selected_store_outfits)} selected ({', '.join(selected_store_outfits[:2])}{'...' if len(selected_store_outfits) > 2 else ''})
    
    ### ⏰ Market Context
    - **Auction Duration**: {auction_duration_hours} hours
    - **Character Age**: {days_since_first_auction} days since first auction
    - **Server Age**: {days_since_server_first_auction} days since server first auction
    
    ### 🔍 Key Value Drivers
    - **Level Impact**: Higher levels exponentially increase value
    - **Skills**: Balanced skills for the vocation are preferred
    - **Assets**: Rare mounts/outfits and high achievement points boost value
    - **Specific Features**: Individual imbuements, charms, and completed quests add targeted value
    - **Market Access**: Transfer availability increases potential buyer pool
    - **Server Popularity**: Established servers typically have higher demand
    
    *This prediction is based on historical auction data and market trends. Actual bids may vary depending on current market conditions, character name appeal, and buyer competition.*
    """
    
    return predicted_bid, explanation

def launch_app():
    """Import Gradio and launch the app."""
    try:
        import gradio as gr
        print(f"Gradio version: {gr.__version__}")
        
        # Create a comprehensive interface with all parameters organized in tabs
        with gr.Blocks(title="Tibia Character Automated Quotation System") as iface:
            gr.Markdown("# 🎮 Tibia Character Automated Quotation System")
            gr.Markdown("Enter your character's attributes to predict the auction price using our XGBoost machine learning model.\n\nSource code: [alvaroinckot/tcaqs](https://github.com/alvaroinckot/tcaqs)")
            
            with gr.Tab("🏆 Basic Info"):
                with gr.Row():
                    with gr.Column():
                        level = gr.Slider(minimum=8, maximum=2000, value=100, step=1, label="Character Level")
                        vocation = gr.Dropdown(choices=VOCATIONS, value="knight", label="Vocation")
                        server = gr.Dropdown(choices=ACTUAL_SERVERS, value="Antica", label="Server")
                        special_char = gr.Checkbox(value=False, label="Name Contains Special Character")
                        
            with gr.Tab("⚔️ Combat Skills"):
                with gr.Row():
                    with gr.Column():
                        axe = gr.Slider(minimum=10, maximum=140, value=80, step=1, label="Axe Fighting")
                        club = gr.Slider(minimum=10, maximum=140, value=10, step=1, label="Club Fighting")
                        sword = gr.Slider(minimum=10, maximum=140, value=10, step=1, label="Sword Fighting")
                        distance = gr.Slider(minimum=10, maximum=140, value=10, step=1, label="Distance Fighting")
                    with gr.Column():
                        magic = gr.Slider(minimum=0, maximum=130, value=5, step=1, label="Magic Level")
                        shielding = gr.Slider(minimum=10, maximum=140, value=80, step=1, label="Shielding")
                        fishing = gr.Slider(minimum=10, maximum=120, value=10, step=1, label="Fishing")
                        fist_fighting = gr.Slider(minimum=10, maximum=140, value=10, step=1, label="Fist Fighting")
                        
            with gr.Tab("💎 Character Assets"):
                with gr.Row():
                    with gr.Column():
                        mounts = gr.Slider(minimum=0, maximum=100, value=5, step=1, label="Mounts")
                        outfits = gr.Slider(minimum=0, maximum=100, value=10, step=1, label="Outfits")
                        gold = gr.Number(value=100000, label="Gold")
                        achievements = gr.Slider(minimum=0, maximum=2000, value=100, step=1, label="Achievement Points")
                        transfer_available = gr.Checkbox(value=False, label="Transfer Available")
                        
            with gr.Tab("✨ Charm System"):
                with gr.Row():
                    with gr.Column():
                        charms = gr.Slider(minimum=0, maximum=20, value=0, step=1, label="Charms")
                        charm_points_available = gr.Slider(minimum=0, maximum=10000, value=1000, step=1, label="Available Charm Points")
                        charm_points_spent = gr.Slider(minimum=0, maximum=20000, value=500, step=1, label="Spent Charm Points")
                        charm_expansion = gr.Checkbox(value=False, label="Charm Expansion")
                        
            with gr.Tab("🎯 Task System"):
                with gr.Row():
                    with gr.Column():
                        hunting_task_points = gr.Slider(minimum=0, maximum=15000, value=100, step=1, label="Hunting Task Points")
                        permanent_prey_slot = gr.Slider(minimum=0, maximum=5, value=0, step=1, label="Permanent Prey Task Slots")
                        permanent_hunt_slot = gr.Slider(minimum=0, maximum=5, value=0, step=1, label="Permanent Hunt Task Slots")
                        prey_wildcards = gr.Slider(minimum=0, maximum=100, value=0, step=1, label="Prey Wildcards")
                        
            with gr.Tab("👥 Hirelings"):
                with gr.Row():
                    with gr.Column():
                        hirelings = gr.Slider(minimum=0, maximum=10, value=0, step=1, label="Hirelings")
                        hirelings_jobs = gr.Slider(minimum=0, maximum=50, value=0, step=1, label="Hireling Jobs")
                        hirelings_outfits = gr.Slider(minimum=0, maximum=20, value=0, step=1, label="Hireling Outfits")
                        
            with gr.Tab("⚡ Imbuements"):
                with gr.Row():
                    with gr.Column():
                        imbuements = gr.Slider(minimum=0, maximum=50, value=0, step=1, label="Imbuements")
                        
            with gr.Tab("🔥 Specific Imbuements"):
                gr.Markdown("### Select which specific imbuements your character has:")
                selected_imbuements = gr.CheckboxGroup(
                    choices=[
                        'imbuements_Powerful_Bash', 'imbuements_Powerful_Blockade', 'imbuements_Powerful_Chop',
                        'imbuements_Powerful_Cloud_Fabric', 'imbuements_Powerful_Demon_Presence', 'imbuements_Powerful_Dragon_Hide',
                        'imbuements_Powerful_Electrify', 'imbuements_Powerful_Epiphany', 'imbuements_Powerful_Featherweight',
                        'imbuements_Powerful_Frost', 'imbuements_Powerful_Lich_Shroud', 'imbuements_Powerful_Precision',
                        'imbuements_Powerful_Quara_Scale', 'imbuements_Powerful_Reap', 'imbuements_Powerful_Scorch',
                        'imbuements_Powerful_Slash', 'imbuements_Powerful_Snake_Skin', 'imbuements_Powerful_Strike',
                        'imbuements_Powerful_Swiftness', 'imbuements_Powerful_Vampirism', 'imbuements_Powerful_Venom',
                        'imbuements_Powerful_Vibrancy', 'imbuements_Powerful_Void'
                    ],
                    value=[],
                    label="Imbuements"
                )
                
            with gr.Tab("✨ Specific Charms"):
                gr.Markdown("### Select which specific charms your character has unlocked:")
                selected_charms_features = gr.CheckboxGroup(
                    choices=[
                        'charms_Adrenaline_Burst', 'charms_Bless', 'charms_Cleanse', 'charms_Cripple', 'charms_Curse',
                        'charms_Divine_Wrath', 'charms_Dodge', 'charms_Enflame', 'charms_Freeze', 'charms_Gut',
                        'charms_Low_Blow', 'charms_Numb', 'charms_Parry', 'charms_Poison', 'charms_Scavenge',
                        'charms_Vampiric_Embrace', 'charms_Voids_Call', 'charms_Wound', 'charms_Zap'
                    ],
                    value=[],
                    label="Charms"
                )
                
            with gr.Tab("📜 Quest Lines"):
                gr.Markdown("### Select which quest lines your character has completed:")
                selected_quest_lines = gr.CheckboxGroup(
                    choices=[
                        'quest_lines_A_Fathers_Burden', 'quest_lines_An_Uneasy_Alliance', 'quest_lines_Blood_Brothers',
                        'quest_lines_Child_of_Destiny', 'quest_lines_Children_of_the_Revolution', 'quest_lines_Dark_Trails',
                        'quest_lines_Dawnport', 'quest_lines_Ferumbras_Ascendant', 'quest_lines_Heart_of_Destruction',
                        'quest_lines_Hero_of_Rathleton', 'quest_lines_Hot_Cuisine', 'quest_lines_In_Service_Of_Yalahar',
                        'quest_lines_Kissing_a_Pig', 'quest_lines_Primal_Ordeal', 'quest_lines_Sea_of_Light',
                        'quest_lines_Shadows_of_Yalahar', 'quest_lines_Soul_War', 'quest_lines_The_Ancient_Tombs',
                        'quest_lines_The_Ape_City', 'quest_lines_The_Beginning', 'quest_lines_The_Desert_Dungeon',
                        'quest_lines_The_Djinn_War_Efreet_Faction', 'quest_lines_The_Djinn_War_Marid_Faction',
                        'quest_lines_The_Explorer_Society', 'quest_lines_The_Gravedigger_of_Drefia', 'quest_lines_The_Ice_Islands',
                        'quest_lines_The_Inquisition', 'quest_lines_The_Isle_Of_Evil', 'quest_lines_The_New_Frontier',
                        'quest_lines_The_Outlaw_Camp', 'quest_lines_The_Paradox_Tower', 'quest_lines_The_Pits_of_Inferno',
                        'quest_lines_The_Postman_Missions', 'quest_lines_The_Queen_of_the_Banshees', 'quest_lines_The_Rookie_Guard',
                        'quest_lines_The_Scatterbrained_Sorcerer', 'quest_lines_The_Shattered_Isles', 'quest_lines_The_Thieves_Guild',
                        'quest_lines_The_White_Raven_Monastery', 'quest_lines_Twenty_Miles_Beneath_The_Sea',
                        'quest_lines_Unnatural_Selection', 'quest_lines_What_a_foolish_Quest', 'quest_lines_Wrath_of_the_Emperor'
                    ],
                    value=[],
                    label="Quest Lines"
                )
                
            with gr.Tab("🐎 Specific Mounts"):
                gr.Markdown("### Select which specific mounts your character owns:")
                selected_mounts_features = gr.CheckboxGroup(
                    choices=[
                        'mounts_Antelope', 'mounts_Black_Sheep', 'mounts_Blazebringer', 'mounts_Blue_Rolling_Barrel',
                        'mounts_Crystal_Wolf', 'mounts_Donkey', 'mounts_Dragonling', 'mounts_Draptor',
                        'mounts_Dromedary', 'mounts_Giant_Beaver', 'mounts_Gnarlhound', 'mounts_Gryphon',
                        'mounts_Hibernal_Moth', 'mounts_Ironblight', 'mounts_Kingly_Deer', 'mounts_Lady_Bug',
                        'mounts_Magma_Crawler', 'mounts_Manta_Ray', 'mounts_Midnight_Panther', 'mounts_Noble_Lion',
                        'mounts_Racing_Bird', 'mounts_Rapid_Boar', 'mounts_Scorpion_King', 'mounts_Stampor',
                        'mounts_Stone_Rhino', 'mounts_Tamed_Panda', 'mounts_Tiger_Slug', 'mounts_Titanica',
                        'mounts_Undead_Cavebear', 'mounts_War_Bear', 'mounts_War_Horse', 'mounts_Water_Buffalo',
                        'mounts_White_Lion', 'mounts_Widow_Queen'
                    ],
                    value=[],
                    label="Regular Mounts"
                )
                
            with gr.Tab("🏪 Store Mounts"):
                gr.Markdown("### Select which store mounts your character owns:")
                selected_store_mounts = gr.CheckboxGroup(
                    choices=[
                        'store_mounts_Arctic_Unicorn', 'store_mounts_Armoured_War_Horse', 'store_mounts_Battle_Badger',
                        'store_mounts_Black_Stag', 'store_mounts_Blazing_Unicorn', 'store_mounts_Crystal_Wolf',
                        'store_mounts_Desert_King', 'store_mounts_Doombringer', 'store_mounts_Emerald_Raven',
                        'store_mounts_Festive_Mammoth', 'store_mounts_Flying_Divan', 'store_mounts_Golden_Dragonfly',
                        'store_mounts_Highland_Yak', 'store_mounts_Magic_Carpet', 'store_mounts_Midnight_Panther',
                        'store_mounts_Prismatic_Unicorn', 'store_mounts_Shadow_Draptor', 'store_mounts_Winter_King'
                    ],
                    value=[],
                    label="Store Mounts (Popular)"
                )
                
            with gr.Tab("👔 Store Outfits"):
                gr.Markdown("### Select which store outfits your character owns:")
                selected_store_outfits = gr.CheckboxGroup(
                    choices=[
                        'store_outfits_Beastmaster_base_and_addon_1_and_addon_2', 'store_outfits_Champion_base_and_addon_1_and_addon_2',
                        'store_outfits_Chaos_Acolyte_base_and_addon_1_and_addon_2', 'store_outfits_Conjurer_base_and_addon_1_and_addon_2',
                        'store_outfits_Death_Herald_base_and_addon_1_and_addon_2', 'store_outfits_Dragon_Knight_base_and_addon_1_and_addon_2',
                        'store_outfits_Entrepreneur_base_and_addon_1_and_addon_2', 'store_outfits_Evoker_base_and_addon_1_and_addon_2',
                        'store_outfits_Forest_Warden_base_and_addon_1_and_addon_2', 'store_outfits_Grove_Keeper_base_and_addon_1_and_addon_2',
                        'store_outfits_Lupine_Warden_base_and_addon_1_and_addon_2', 'store_outfits_Mercenary_base_and_addon_1_and_addon_2',
                        'store_outfits_Pharaoh_base_and_addon_1_and_addon_2', 'store_outfits_Philosopher_base_and_addon_1_and_addon_2',
                        'store_outfits_Ranger_base_and_addon_1_and_addon_2', 'store_outfits_Sea_Dog_base_and_addon_1_and_addon_2',
                        'store_outfits_Siege_Master_base_and_addon_1_and_addon_2', 'store_outfits_Spirit_Caller_base_and_addon_1_and_addon_2'
                    ],
                    value=[],
                    label="Store Outfits (Popular)"
                )
                        
            with gr.Tab("⏰ Auction Settings"):
                with gr.Row():
                    with gr.Column():
                        auction_duration_hours = gr.Slider(minimum=24, maximum=168, value=48, step=1, label="Auction Duration (hours)")
                        days_since_first_auction = gr.Slider(minimum=0, maximum=3000, value=365, step=1, label="Days Since First Auction")
                        days_since_server_first_auction = gr.Slider(minimum=0, maximum=2000, value=200, step=1, label="Days Since Server First Auction")
            
            # Prediction button and output
            with gr.Row():
                predict_btn = gr.Button("🔮 Predict Character Value", variant="primary", scale=2)
            
            with gr.Row():
                with gr.Column():
                    predicted_bid = gr.Number(label="Predicted Bid (Tibia Coins)", precision=0)
                with gr.Column():
                    explanation = gr.Markdown(label="Prediction Explanation")
            
            # Connect the predict button to the function
            predict_btn.click(
                fn=predict_from_interface,
                inputs=[
                    # Basic Info
                    level, vocation, server, special_char,
                    # Combat Skills  
                    axe, club, sword, distance, magic, shielding, fishing, fist_fighting,
                    # Character Assets
                    mounts, outfits, gold, achievements, transfer_available,
                    # Charm System
                    charms, charm_points_available, charm_points_spent, charm_expansion,
                    # Task System
                    hunting_task_points, permanent_prey_slot, permanent_hunt_slot, prey_wildcards,
                    # Hirelings
                    hirelings, hirelings_jobs, hirelings_outfits,
                    # Imbuements
                    imbuements,
                    # Auction Settings
                    auction_duration_hours, days_since_first_auction, days_since_server_first_auction,
                    # Boolean feature selections
                    selected_imbuements, selected_charms_features, selected_quest_lines,
                    selected_mounts_features, selected_store_mounts, selected_store_outfits
                ],
                outputs=[predicted_bid, explanation]
            )
            
            # Add some examples
            gr.Examples(
                examples=[
                    # [level, vocation, server, special_char, axe, club, sword, distance, magic, shielding, fishing, fist_fighting, mounts, outfits, gold, achievements, transfer_available, charms, charm_points_available, charm_points_spent, charm_expansion, hunting_task_points, permanent_prey_slot, permanent_hunt_slot, prey_wildcards, hirelings, hirelings_jobs, hirelings_outfits, imbuements, auction_duration_hours, days_since_first_auction, days_since_server_first_auction, selected_imbuements, selected_charms_features, selected_quest_lines, selected_mounts_features, selected_store_mounts, selected_store_outfits]
                    [100, "knight", "Antica", False, 80, 10, 10, 10, 5, 80, 10, 10, 5, 10, 100000, 100, False, 0, 1000, 500, False, 100, 0, 0, 0, 0, 0, 0, 0, 48, 365, 200, [], [], [], [], [], []],
                    [500, "paladin", "Secura", False, 15, 15, 15, 120, 30, 100, 15, 15, 20, 30, 5000000, 800, True, 10, 5000, 3000, True, 2000, 2, 1, 50, 3, 15, 8, 25, 48, 1000, 500, ['imbuements_Powerful_Vampirism', 'imbuements_Powerful_Strike'], ['charms_Divine_Wrath', 'charms_Dodge'], ['quest_lines_The_Inquisition', 'quest_lines_Wrath_of_the_Emperor'], ['mounts_War_Horse', 'mounts_Midnight_Panther'], ['store_mounts_Armoured_War_Horse'], ['store_outfits_Champion_base_and_addon_1_and_addon_2']],
                    [250, "sorcerer", "Antica", False, 10, 10, 10, 10, 80, 20, 10, 10, 15, 25, 1000000, 400, False, 5, 2000, 1500, False, 500, 1, 0, 20, 1, 5, 3, 12, 72, 600, 300, ['imbuements_Powerful_Epiphany'], ['charms_Enflame'], ['quest_lines_Ferumbras_Ascendant'], ['mounts_Crystal_Wolf'], [], ['store_outfits_Conjurer_base_and_addon_1_and_addon_2']]
                ],
                inputs=[level, vocation, server, special_char, axe, club, sword, distance, magic, shielding, fishing, fist_fighting, mounts, outfits, gold, achievements, transfer_available, charms, charm_points_available, charm_points_spent, charm_expansion, hunting_task_points, permanent_prey_slot, permanent_hunt_slot, prey_wildcards, hirelings, hirelings_jobs, hirelings_outfits, imbuements, auction_duration_hours, days_since_first_auction, days_since_server_first_auction, selected_imbuements, selected_charms_features, selected_quest_lines, selected_mounts_features, selected_store_mounts, selected_store_outfits],
                label="Example Characters"
            )
            
            gr.Markdown("""
            ### 📊 About This Predictor
            
            This tool uses an **XGBoost machine learning model** trained on historical Tibia character auction data to predict bid values. 
            
            **Key Features:**
            - ⚔️ **Combat Skills**: All fighting skills and magic level
            - 💎 **Assets**: Mounts, outfits, gold, and achievements  
            - ✨ **Advanced Systems**: Charms, hunting tasks, hirelings, and imbuements
            - 🔥 **Specific Features**: Individual imbuements, charms, quest lines, mounts, and store items
            - ⏰ **Market Timing**: Auction duration and historical patterns
            
            **New Boolean Feature Selection:**
            - 🔥 **Specific Imbuements**: Select exactly which imbuements your character has
            - ✨ **Specific Charms**: Choose individual unlocked charms
            - 📜 **Quest Lines**: Mark completed quest lines that add value
            - 🐎 **Specific Mounts**: Select individual mounts owned
            - 🏪 **Store Items**: Choose store mounts and outfits
            
            **Tips for Better Predictions:**
            - Higher level characters typically have higher values
            - Rare mounts and outfits significantly increase value
            - Specific imbuements and charms can boost value for certain vocations
            - Completed quest lines show character progression
            - Transfer availability affects market reach
            - Server popularity influences demand
            
            *Predictions are estimates based on historical data and may not reflect current market conditions.*
            """)
        
        # Launch the interface with deployment-friendly settings
        iface.launch(
            share=False,  # Don't create public link automatically
            server_name="0.0.0.0",  # Make it accessible from any IP
            server_port=7860,  # Use standard Gradio port
            show_error=True,  # Show errors for debugging
            quiet=False  # Show startup logs
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