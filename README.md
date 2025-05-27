# Critter and Flora - Species Recognition System

## Project Overview

Critter and Flora is an advanced species identification and information system that uses image recognition technology to identify birds, insects, plants, and animals from user-uploaded images. The system provides detailed information about identified species, including scientific name, habitat, diet, lifespan, physical characteristics, and interesting facts.

## Features

- **Multi-Species Recognition**: Upload an image of a bird, insect, plant, or animal and get an accurate identification using machine learning models
- **Comprehensive Species Database**: Access detailed information for hundreds of species
- **Educational Content**: Learn about habitats, behaviors, and interesting facts
- **User-Friendly Interface**: Intuitive design for easy navigation and use
- **Responsive Design**: Works across desktop and mobile devices

## Technical Architecture

The application is built using:
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python, Flask
- **Database**: MySQL
- **ML Models**: TensorFlow for species recognition
- **Authentication**: Secure user login and registration system

## Data Sources

The species information database includes comprehensive details compiled from:
- Scientific journals and research
- Field guides and manuals
- Expert-verified descriptions and measurements
- High-quality images for reference and identification

## Installation and Setup

### Prerequisites
- Python 3.8+
- MySQL
- TensorFlow 2.x
- Flask
- Required Python packages (listed in requirements.txt)

### Installation Steps

1. Clone the repository:
```
git clone https://github.com/Lokesh071/Critter-and-Flora.git
cd Critter-and-Flora
```

2. Set up a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```
pip install -r requirements.txt
```

4. Create a .env file using the .env.template file
   - Update the database configuration
   - Add your email credentials for password reset functionality

5. Configure the database:
   - Create a MySQL database named nature_auth
   - Run the database setup script

6. Start the application:
```
python app.py
```

7. Access the application in your browser at `http://localhost:5000`

## Usage

1. **Register/Login**: Create an account or log in to an existing account
2. **Upload Image**: Upload an image for identification
3. **View Results**: Review the identification results and detailed species information
4. **Explore Database**: Browse the species database by category, habitat, or other filters

## Project Structure

```
Critter-and-Flora/
├── app.py                 # Main Flask application file
├── requirements.txt       # Required Python packages
├── model/                 # ML models and data files
│   ├── model_utils.py     # Model utility functions
│   ├── class_names.txt    # Species class names
│   └── *_info.json        # Detailed species information
├── static/                # Static files (CSS, JS, images)
│   ├── css/
│   ├── js/
│   └── assets/
├── templates/             # HTML templates
└── uploads/               # User-uploaded images for processing
```
OUTPUT SCREENS:
Screenshot 2025-05-05 144117

image

image

image

image

image

image

image
## License

This project is licensed under the MIT License.

## Acknowledgements

- Species data sourced from reputable databases
- The TensorFlow and machine learning communities for providing tools and resources for image recognition
