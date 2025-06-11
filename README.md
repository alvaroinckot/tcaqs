---
title: TCAQS_-_Tibia_Character_Price_Predictor
app_file: app.py
sdk: gradio
sdk_version: 5.31.0
---
# 🎮 TCAQS - Tibia Character Automated Quotation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Web App](https://img.shields.io/badge/Web-Gradio-green.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **A sophisticated machine learning system that predicts Tibia character auction prices with 93.49% accuracy using advanced regression models and comprehensive feature engineering.**

## 🎯 Project Overview

TCAQS is an end-to-end machine learning solution designed to solve a real-world gaming economy problem. The system predicts auction prices for Tibia characters by analyzing 400+ features including combat skills, rare items, achievements, and market dynamics.

**Key Achievements:**
- 🎯 **93.49% R² accuracy** on test data
- 🔍 **400+ engineered features** from character profiles
- 📊 **650,000+ character dataset** from 2022 auction data
- 🚀 **Production-ready web application** with Gradio
- 📊 **Comprehensive data pipeline** from scraping to deployment

## 🏗️ System Architecture

```
📦 Data Collection (Web Scraping)
    ↓
🔧 Feature Engineering (400+ features)
    ↓
🤖 Model Training (XGBoost Regression)
    ↓
🌐 Web Application (Gradio Interface)
    ↓
☁️ Deployment Ready (Gradio Spaces)
```

## 📊 Model Performance

Our XGBoost model demonstrates excellent performance across multiple metrics:

| Model | Split | MSE | RMSE | MAE | MAPE | R² | Explained Variance |
|-------|-------|-----|------|-----|------|----|--------------------|
| **XGBoost v3** | Training | 49,717.34 | 222.97 | 149.20 | 0.3372 | **0.9954** | 0.9954 |
| **XGBoost v3** | Testing | 725,773.25 | 851.92 | 358.36 | 0.4243 | **0.9349** | 0.9349 |

### 🎯 Key Performance Insights
- **R² Score of 0.9349**: Explains 93.49% of price variance
- **Low Overfitting**: Training R² (0.9954) vs Testing R² (0.9349) shows good generalization
- **MAPE of 42.43%**: Reasonable error rate for gaming asset predictions
- **Robust Predictions**: Model handles outliers and edge cases effectively

## 🛠️ Technical Stack

### **Machine Learning & Data Science**
- **XGBoost**: Primary regression model
- **scikit-learn**: Preprocessing and model evaluation
- **pandas & numpy**: Data manipulation and analysis
- **Feature Engineering**: 400+ derived features

### **Web Development**
- **Gradio**: Interactive web interface
- **Python**: Backend logic and model serving

### **Data Management**
- **SQLite**: Character database storage
- **BeautifulSoup**: Web scraping engine
- **JSON**: Configuration and metadata

## 🚀 Features

### **🎮 Comprehensive Character Analysis**
- **Combat Skills**: All fighting skills, magic level, shielding
- **Character Assets**: Mounts, outfits, gold, achievements
- **Advanced Systems**: Charms, imbuements, hunting tasks, hirelings
- **Quest Progress**: 40+ major quest lines tracking
- **Market Factors**: Server type, transfer availability, auction timing

### **🔍 Advanced Feature Engineering**
- **Temporal Features**: Days since character/server creation
- **Boolean Encodings**: 300+ specific items and achievements
- **Categorical Processing**: Server locations, PvP types, vocations
- **Interaction Features**: Skill combinations and ratios

### **🌐 Production-Ready Web Interface**
- **Intuitive Tabbed Design**: Organized input categories
- **Real-time Predictions**: Instant price estimates
- **Detailed Explanations**: Comprehensive result breakdowns
- **Example Characters**: Pre-loaded test cases


## 🔧 Installation & Setup

### **Prerequisites**
- Python 3.8+
- 4GB+ RAM (for model loading)
- Internet connection (for web interface)

### **Quick Start**

1. **Clone the repository**
   ```bash
   git clone https://github.com/alvaroinckot/tcaqs.git
   cd tcaqs
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the web application**
   ```bash
   python app.py
   ```

4. **Access the interface**
   - Local: http://localhost:7860
   - The terminal will display additional access URLs

### **Alternative: Direct Deployment**
```bash
gradio deploy  # Deploy to Gradio Spaces
```

## 📈 Data Science Methodology

### **1. Data Collection**
- **Web Scraping**: Automated extraction from Tibia auction pages (2022)
- **Data Volume**: ~650,000 character profiles
- **Time Period**: Historical auction data from 2022
- **Quality Assurance**: Automated validation and cleaning

### **2. Feature Engineering**
- **Raw Features**: 30+ direct character attributes
- **Derived Features**: 370+ engineered features
- **Categorical Encoding**: One-hot encoding for 100+ categories
- **Temporal Features**: Time-based market dynamics

### **3. Model Development**
- **Algorithm Selection**: XGBoost chosen after comparing multiple models
- **Hyperparameter Tuning**: Grid search optimization
- **Cross-Validation**: K-fold validation for robust evaluation
- **Feature Selection**: Importance-based feature pruning

### **4. Model Evaluation**
- **Multiple Metrics**: MSE, RMSE, MAE, MAPE, R²
- **Train/Test Split**: 80/20 stratified split
- **Outlier Analysis**: Robust performance on edge cases
- **Feature Importance**: Top predictors identified and analyzed


## 🚀 Deployment

### **Local Development**
```bash
python app.py  # Starts local server on port 7860
```

## 📋 Development Roadmap

### **✅ Completed**
- [x] Data collection and preprocessing pipeline
- [x] Feature engineering (400+ features)
- [x] XGBoost model training and optimization
- [x] Web interface development
- [x] Deployment preparation

### **🔄 In Progress**
- [ ] Real-time data updates
- [ ] Advanced visualization dashboard
- [ ] API endpoint development

### **📅 Future Enhancements**
- [ ] Update dataset with post-2022 auction data
- [ ] Expand training data beyond 650k characters
- [ ] Integrate new Tibia features released after 2022:
  - [ ] New mounts and outfits from recent updates
  - [ ] Updated charm system changes
  - [ ] New quest lines and bosses
  - [ ] Bosstiary tracking system
  - [ ] Updated imbuement system
- [ ] Deep learning model comparison
- [ ] Multi-server price analysis
- [ ] Historical trend predictions
- [ ] Mobile-responsive interface
- [ ] API documentation and SDK

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### **Development Setup**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Exevo Pan Project**: Special thanks to the [Exevo Pan](https://exevopan.com) team for their invaluable contribution to the Tibia community and inspiration for character auction analysis
- **Tibia Community**: For the rich gaming ecosystem that inspired this project
- **CipSoft**: For creating the Tibia MMORPG
- **Open Source Libraries**: XGBoost, Gradio, scikit-learn, and pandas teams
- **Data Science Community**: For methodologies and best practices

## 📞 Contact

**Alvaro** - [@alvaroinckot](https://github.com/alvaroinckot)

Project Link: [https://github.com/alvaroinckot/tcaqs](https://github.com/alvaroinckot/tcaqs)

---

⭐ **Star this repository if you found it helpful!** ⭐

