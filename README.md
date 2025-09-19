# Deepfake Detection App

A web application that uses machine learning to detect deepfake images with high accuracy.

## Features

- 🎯 **High Accuracy**: Achieves 85%+ accuracy in deepfake detection
- 🌐 **Web Interface**: User-friendly Gradio web interface
- 📱 **Responsive Design**: Works on desktop and mobile devices
- 🔍 **Detailed Analysis**: Provides confidence scores and visual analysis
- ⚡ **Fast Processing**: Quick image analysis using optimized models

## Technology Stack

- **Backend**: Python, TensorFlow/Keras
- **Frontend**: Gradio
- **Model**: MobileNetV2 with transfer learning
- **Image Processing**: OpenCV, PIL

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deepfake-detection-app.git
cd deepfake-detection-app
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the trained model:
   - Place your trained model file as `deepfake_final_model.keras` in the root directory
   - Or train your own model using the provided training scripts

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser and go to:
```
http://localhost:7860
```

3. Upload an image to analyze for deepfake detection

## Training Your Own Model

The repository includes training scripts for creating custom deepfake detection models:

- `deepfake_model.py` - Basic training script
- `deepfake_model_working.py` - Optimized training script
- `deepfake_model_final.py` - Final production-ready script

## Model Performance

- **Accuracy**: 85%+ on validation set
- **Architecture**: MobileNetV2 with custom classifier
- **Input Size**: 224x224 pixels
- **Classes**: Real vs Fake images

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for educational and research purposes. Always verify important information through multiple sources and use additional verification methods for critical decisions.
# deepfake-detection-app
