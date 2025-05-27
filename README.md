# TCAQS - Tibia Character Automated Quotation System

A machine learning system that predicts the auction price of Tibia characters based on their attributes.

## Features

- Character price prediction using XGBoost regression model
- Web interface for easy prediction using Gradio
- Data extraction from HTML auction pages
- Comprehensive analysis and visualization of character data

## Setup and Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r dependencies.txt
   ```

## Running the Web Interface

To launch the character price prediction web interface:

```bash
cd scripts
./ui.py
```

This will start a Gradio web server on port 7860. You can access it at:
- http://localhost:7860 (local access)
- A public shareable link will also be displayed in the terminal

