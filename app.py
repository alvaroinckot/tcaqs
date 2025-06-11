#!/usr/bin/env python3
"""
Main entry point for the Tibia Character Automated Quotation System (TCAQS)
Gradio deployment file
"""

import sys
import os

# Add the scripts directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

def main():
    """Main function to launch the app with error handling."""
    try:
        # Check if model exists
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'xgboost_character_bid_model_v3.pkl')
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}")
            print("The app will try to find the model in alternative locations.")
        
        # Import and launch the UI
        from ui import launch_app
        print("Starting TCAQS Gradio application...")
        launch_app()
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please make sure all dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
