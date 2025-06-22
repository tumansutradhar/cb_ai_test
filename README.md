# AI-Based Road Debris and Garbage Detection System

## Overview

This project implements an AI-powered deep learning solution to detect debris, garbage, and construction materials that are temporarily or permanently placed on roads in India. The system combines a Python-based computer vision backend with a modern React TypeScript frontend to provide real-time detection capabilities and an intuitive web interface for monitoring and alert management.

## Problem Statement

In India, it's common for construction materials such as bricks, sand, soil, and stones to be temporarily stored on roadsides during building or wall construction projects. Additionally, bulk garbage is often disposed of or stored along roadways for extended periods. These obstructions:

- Create safety hazards and increase accident risks
- Make it difficult for pedestrians and vehicles to navigate roads
- Contribute to urban and rural cleanliness issues
- Require timely intervention from local authorities

## Solution

Our AI-based detection system uses computer vision and deep learning to:

1. **Automatically detect** debris, garbage, and construction materials on roads
2. **Classify** different types of obstructions (construction debris, garbage, etc.)
3. **Provide a web interface** for real-time monitoring and visualization
4. **Enable alert management** through an intuitive dashboard
5. **Support batch processing** for large-scale monitoring

## Project Structure

```
cd_ai/
├── backend/
│   ├── app.py                 # Main Flask/FastAPI application
│   ├── detector.py            # Core detection logic
│   ├── best.pt               # Trained model weights (YOLO)
│   ├── road.jpg              # Sample test image
│   ├── sample.mp4            # Sample test video
│   ├── 123.mp4               # Test video file
│   └── static/
│       └── uploads/          # Processed media files
│           ├── 123_detected.mp4
│           ├── sample_detected.mp4
│           └── ...
├── frontend/
│   ├── src/
│   │   ├── App.tsx           # Main React application
│   │   ├── main.tsx          # Application entry point
│   │   ├── index.css         # Global styles with Tailwind
│   │   ├── components/       # Reusable React components
│   │   │   ├── Loader.tsx
│   │   │   └── Navbar.tsx
│   │   └── pages/            # Application pages
│   ├── package.json          # Frontend dependencies
│   ├── vite.config.ts        # Vite configuration
│   ├── tailwind.config.js    # Tailwind CSS configuration
│   ├── tsconfig.json         # TypeScript configuration
│   └── index.html            # HTML template
└── README.md                 # This file
```

## Features

- ✅ **Real-time debris and garbage detection** using YOLO model
- ✅ **Multi-class object detection** (construction materials, garbage)
- ✅ **Custom dataset** trained on Indian road conditions(rural and semi-urban areas)
- ✅ **Modern React TypeScript frontend** with Vite
- ✅ **Responsive design** with Tailwind CSS
- ✅ **Video and image processing** capabilities
- ✅ **File upload and processing** interface
- ✅ **Real-time detection visualization**
- ✅ **Scalable architecture** for deployment

## Technical Stack

### Backend
- **Python** with computer vision libraries
- **YOLO** model for object detection ([best.pt](backend/best.pt))
- **Flask/FastAPI** for REST API
- **OpenCV** for image/video processing
- **PyTorch** for deep learning inference

### Frontend
- **React 18** with TypeScript
- **Vite** for fast development and building
- **Tailwind CSS** for responsive styling
- **Lucide React** for icons
- **React Router DOM** for navigation
- **ESLint** for code quality

## Installation & Setup

### Prerequisites
- Python 3.8+
- Node.js 16+
- GPU recommended for optimal performance

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/ranadebsaha/CD_AI
cd cd_ai/backend

# Install Python dependencies
pip install -r requirements.txt

# Run the backend server
python app.py
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

## Usage

### Web Interface
1. Start both backend and frontend servers
2. Navigate to `http://localhost:5173` (or configured port)
3. Use the web interface to:
   - Upload images or videos for detection
   - View real-time detection results
   - Monitor processed files
   - Access detection history

### API Endpoints
```bash
# Upload and process image/video
POST /api/detect
Content-Type: multipart/form-data

# Get processed results
GET /api/results

# Health check
GET /api/health
```

## Model Information

The system uses a custom-trained YOLO model ([best.pt](backend/best.pt)) specifically optimized for detecting:

- **Construction debris**: Bricks, sand, stones, building materials
- **Bulk garbage**: Waste materials, trash piles
- **Mixed obstructions**: Combined debris types

### Model Performance
- Trained on Indian road conditions
- Optimized for various lighting conditions
- Real-time processing capabilities
- High accuracy for debris classification

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Test both frontend and backend
5. Commit changes (`git commit -am 'Add feature'`)
6. Push to branch (`git push origin feature/improvement`)
7. Create Pull Request

## Future Enhancements

### Technical Improvements
- [ ] Mobile app development (React Native)
- [ ] Advanced analytics dashboard
- [ ] User authentication and roles
- [ ] Database integration for detection history
- [ ] GPS integration for location tracking

### AI/ML Enhancements
- [ ] Model accuracy improvements
- [ ] Edge device deployment
- [ ] Multi-model ensemble
- [ ] Custom training pipeline
- [ ] Data augmentation techniques

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Roboflow** for dataset management platform
- **YOLO** community for object detection framework
- **React** and **Vite** communities for frontend tools
- **Tailwind CSS** for styling framework
- Local authorities and communities for data collection support

---

**Note**: This project is developed to serve society by improving road safety conditions in India. We encourage responsible use and welcome community participation in making roads safer for everyone.