# Nature - Bird Species Recognition System

## Project Overview

Nature is an advanced bird species identification and information system that uses image recognition technology to identify birds from user-uploaded images. The system provides detailed information about identified bird species, including scientific name, habitat, diet, lifespan, physical characteristics, and interesting facts.

## Features

- **Bird Species Recognition**: Upload an image of a bird and get an accurate identification using machine learning models
- **Comprehensive Bird Database**: Access detailed information for hundreds of bird species
- **Educational Content**: Learn about bird habitats, behaviors, and interesting facts
- **User-Friendly Interface**: Intuitive design for easy navigation and use
- **Responsive Design**: Works across desktop and mobile devices

## Technical Architecture

The application is built using:
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python, Flask
- **Database**: MySQL
- **ML Models**: TensorFlow for bird species recognition
- **Authentication**: Secure user login and registration system

## Data Sources

The bird information database includes comprehensive details on bird species compiled from:
- Scientific journals and ornithological research
- Bird watching guides and field manuals
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
git clone https://github.com/Lokesh071/Nature.git
cd Nature
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

4. Configure the database:
   - Create a MySQL database
   - Update the database configuration in the `app.py` file
   - Run the database setup script:
   ```
   python db_setup.py
   ```

5. Start the application:
```
python app.py
```

6. Access the application in your browser at `http://localhost:5000`

## Usage

1. **Register/Login**: Create an account or log in to an existing account
2. **Upload Image**: Upload an image of a bird for identification
3. **View Results**: Review the identification results and detailed species information
4. **Explore Database**: Browse the bird species database by category, habitat, or other filters
5. **Learning Resources**: Access educational content about bird watching and ornithology

## Project Structure

```
Nature/
├── app.py                 # Main Flask application file
├── db_setup.py            # Database setup script
├── requirements.txt       # Required Python packages
├── model/                 # ML models and data files
│   ├── bird_model.h5      # Trained TensorFlow model
│   ├── class_names.txt    # Bird species class names
│   └── bird_info_converted.json  # Detailed bird information
├── static/                # Static files (CSS, JS, images)
│   ├── css/
│   ├── js/
│   └── images/
├── templates/             # HTML templates
│   ├── index.html
│   ├── upload.html
│   ├── results.html
│   └── ...
└── uploads/               # User-uploaded images for processing
```

## Contributing

Contributions to the Nature project are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Run tests to ensure functionality
5. Commit your changes (`git commit -m 'Add some feature'`)
6. Push to the branch (`git push origin feature/your-feature-name`)
7. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Bird image data sourced from reputable ornithological databases
- Special thanks to the ornithologists and bird enthusiasts who contributed to the accuracy of our bird information database
- The TensorFlow and machine learning communities for providing tools and resources for image recognition 