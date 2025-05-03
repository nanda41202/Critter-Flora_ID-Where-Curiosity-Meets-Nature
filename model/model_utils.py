import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import urllib.request
import tensorflow as tf
import os
import json
import logging
from tensorflow.keras.models import load_model
from keras.layers import Layer

class CustomLayer(Layer):
    def __init__(self, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return inputs

    def get_config(self):
        return super(CustomLayer, self).get_config()

# Create aliases for common custom layer names
CustomLayer1 = CustomLayer
CustomLayer2 = CustomLayer

# Define the base directory relative to this file's location (model/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define paths relative to the project root (one level up from model/)
PROJECT_ROOT = os.path.join(BASE_DIR, '..')
BIRDS_FOLDER_PATH = os.path.join(PROJECT_ROOT, 'birds')
# Dog model is expected in the same directory as this script (model/)
MODEL_DIR = BASE_DIR

# --- Helper Function for Model Input Shape ---
def get_model_input_shape(model, default_shape=(224, 224)):
    """Gets the input shape (height, width) from a Keras model, or returns a default."""
    try:
        # Handle different ways input_shape might be structured
        if isinstance(model.input_shape, tuple) and len(model.input_shape) >= 3:
            # e.g., (None, height, width, channels)
            return (model.input_shape[1], model.input_shape[2])
        elif isinstance(model.input_shape, list) and len(model.input_shape[0]) >= 3:
             # e.g., [(None, height, width, channels)]
             return (model.input_shape[0][1], model.input_shape[0][2])
        else:
             print(f"Warning: Could not determine input shape from model.input_shape: {model.input_shape}. Using default {default_shape}.")
             return default_shape
    except Exception as e:
        print(f"Warning: Error getting input shape: {e}. Using default {default_shape}.")
        return default_shape

from typing import Optional, Dict, List, Tuple, Union
import numpy as np
import torch
from PIL import Image

# Add this function at the top of the file to check if TensorFlow is available
def is_tensorflow_available():
    """Check if TensorFlow is available."""
    try:
        import tensorflow
        return True
    except ImportError:
        return False

class ModelLoader:
    """Handles loading and prediction for bird species classification and general image classification models."""
    
    def __init__(self):
        """Initialize the model loader and information dictionaries."""
        # Model availability flags
        self.models_loaded = False
        self.using_pytorch = False
        
        # General classification models
        self.model = None
        self.labels = None
        self.general_model = None
        self.general_labels = None
        
        # Bird specific models and data
        self.bird_model = None
        self.bird_labels = []
        self.bird_info = {}
        self.bird_input_shape = (224, 224)
        
        # Dog specific models and data
        self.dog_model = None
        self.dog_labels = []
        self.dog_info = {}
        self.dog_input_shape = (224, 224)
        
        # Flower specific models and data
        self.flower_model = None
        self.flower_labels = []
        self.flower_info = {}
        self.flower_input_shape = (224, 224)
        
        # Plant specific models and data
        self.plant_model = None
        self.plant_labels = []
        self.plant_info = {}
        self.plant_input_shape = (224, 224)
        
        # Insect specific models and data
        self.insect_model = None
        self.insect_labels = {}
        self.insect_info = {}
        self.insect_input_shape = (224, 224)
        
        # Cat specific models and data
        self.cat_model = None
        self.cat_labels = []
        self.cat_input_shape = (224, 224)
        
        # Cat breed information dictionary
        self.cat_info = {
            "persian cat": {
                "scientific_name": "Felis catus",
                "description": "Persian cats are known for their long, luxurious coats and flat faces. They have a sweet, gentle temperament and prefer calm environments. Persians are one of the oldest cat breeds, requiring regular grooming to maintain their beautiful coat.",
                "origin": "Persia (Iran)",
                "lifespan": "12-17 years",
                "weight": "3.5-5.5 kg",
                "height": "25-38 cm",
                "temperament": "Calm, affectionate, and quiet"
            },
            "siamese cat": {
                "scientific_name": "Felis catus",
                "description": "Siamese cats are known for their striking blue almond-shaped eyes and color point coat pattern. They are highly intelligent, vocal, and social cats that form strong bonds with their humans. Siamese are one of the oldest recognized breeds of oriental cats.",
                "origin": "Thailand (formerly Siam)",
                "lifespan": "12-20 years",
                "weight": "2.5-5.5 kg",
                "height": "20-30 cm",
                "temperament": "Active, intelligent, and vocal"
            },
            "egyptian cat": {
                "scientific_name": "Felis catus",
                "description": "Often referring to the Egyptian Mau, these spotted cats are one of the few naturally spotted domestic cat breeds. They are known for their agility and speed, being among the fastest of domestic cats. Egyptian Maus have a strong historical connection to ancient Egypt.",
                "origin": "Egypt",
                "lifespan": "12-15 years",
                "weight": "3-5 kg",
                "height": "20-25 cm",
                "temperament": "Active, agile, and loyal"
            },
            "maine coon": {
                "scientific_name": "Felis catus",
                "description": "Maine Coons are one of the largest domestic cat breeds, known for their tufted ears, bushy tails, and shaggy coats. Despite their imposing size, they are gentle giants with a friendly, playful disposition. They are excellent mousers and adapt well to various environments.",
                "origin": "United States",
                "lifespan": "12-15 years",
                "weight": "5.5-9 kg",
                "height": "25-40 cm",
                "temperament": "Gentle, friendly, and intelligent"
            },
            "british shorthair": {
                "scientific_name": "Felis catus",
                "description": "British Shorthairs are known for their dense, plush coats and round faces with bright copper eyes. They are an easygoing breed with a calm demeanor. Their blue-gray variant is particularly popular, though they come in many colors and patterns.",
                "origin": "Great Britain",
                "lifespan": "12-20 years",
                "weight": "4-8 kg",
                "height": "30-35 cm",
                "temperament": "Easygoing, affectionate, and independent"
            },
            "abyssinian": {
                "scientific_name": "Felis catus",
                "description": "Abyssinians have a distinctive ticked coat pattern, giving them a wild appearance. They are active, intelligent cats that enjoy playing and exploring. Their coat resembles that of wild cats from Africa, with a warm reddish or ruddy brown being the most common color.",
                "origin": "Ethiopia (formerly Abyssinia)",
                "lifespan": "9-15 years",
                "weight": "3-5 kg",
                "height": "20-30 cm",
                "temperament": "Active, playful, and curious"
            },
            "bengal cat": {
                "scientific_name": "Felis catus",
                "description": "Bengal cats are known for their wild appearance with distinctive spots and rosettes on a background of gold, brown, or gray. They are a hybrid breed developed from crossing domestic cats with the Asian leopard cat. Bengals are energetic, playful, and often enjoy water.",
                "origin": "United States",
                "lifespan": "12-16 years",
                "weight": "4-7 kg",
                "height": "25-35 cm",
                "temperament": "Energetic, playful, and intelligent"
            },
            "ragdoll": {
                "scientific_name": "Felis catus",
                "description": "Ragdolls are large, semi-longhaired cats known for their docile and placid temperament. They go limp when picked up, hence their name. With striking blue eyes and color point patterns, they are known for being gentle and affectionate companions.",
                "origin": "United States",
                "lifespan": "12-17 years",
                "weight": "5-9 kg",
                "height": "25-35 cm",
                "temperament": "Docile, calm, and affectionate"
            },
            "tabby cat": {
                "scientific_name": "Felis catus",
                "description": "Tabby refers to a coat pattern rather than a breed, characterized by distinctive stripes, swirls, or spots. This is the most common coat pattern in domestic cats. Tabby patterns include classic (swirled), mackerel (striped), spotted, and ticked varieties.",
                "origin": "Worldwide",
                "lifespan": "12-18 years",
                "weight": "3.5-7 kg",
                "height": "23-30 cm",
                "temperament": "Varies widely"
            },
            "tiger cat": {
                "scientific_name": "Felis catus",
                "description": "Tiger cat typically refers to a tabby cat with distinctive striped markings reminiscent of a tiger. These are domestic cats with a particular coat pattern rather than a specific breed. The striped pattern provides excellent camouflage in the wild.",
                "origin": "Worldwide",
                "lifespan": "12-18 years",
                "weight": "3.5-7 kg",
                "height": "23-30 cm",
                "temperament": "Varies widely"
            },
            "cat": {
                "scientific_name": "Felis catus",
                "description": "Domestic cats are carnivorous mammals from the family Felidae. Known for their agility, stealth, and independence, they have been human companions for thousands of years. Cats communicate through meowing, purring, and body language, and are known for their grooming behavior.",
                "origin": "Middle East/North Africa",
                "lifespan": "12-18 years",
                "weight": "3.5-7 kg",
                "height": "23-30 cm",
                "temperament": "Independent, curious, and affectionate"
            }
        }
        
        # Load all required models
        self.load_models()
        
    def load_models(self) -> bool:
        """Load all necessary models for inference."""
        # Define custom objects for Keras models
        custom_objects = {
            'CustomLayer': CustomLayer,
            'CustomLayer1': CustomLayer,
            'CustomLayer2': CustomLayer
        }
        
        print("\n\n🔄 LOADING MODELS")
        print(f"📂 Model directory: {MODEL_DIR}")
        print(f"📂 Bird folder path: {BIRDS_FOLDER_PATH}")
        
        try:
            # 1. Load General image classification model
            print("🔄 Loading general image classification model...")
            
            # Check for model file
            if os.path.exists(os.path.join(MODEL_DIR, "resnet50_model.h5")):
                try:
                    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
                    from tensorflow.keras.applications import imagenet_utils
                    
                    # Load model with pre-trained weights
                    self.model = ResNet50(weights='imagenet')
                    
                    # Load labels
                    imagenet_labels_path = os.path.join(MODEL_DIR, 'imagenet_class_index.json')
                    
                    if os.path.exists(imagenet_labels_path):
                        with open(imagenet_labels_path, "r") as f:
                            indices = json.load(f)
                            self.labels = [indices[str(i)][1] for i in range(len(indices))]
                    else:
                        # Fallback to web API for labels
                        response = requests.get("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json")
                        indices = response.json()
                        self.labels = [indices[str(i)][1] for i in range(len(indices))]
                    
                    print(f"✅ Loaded TensorFlow ResNet50 model ({len(self.labels)} classes)")
                    self.using_pytorch = False
                except Exception as e:
                    print(f"❌ Error loading TensorFlow model: {str(e)}")
                    self.model = None
                    self.labels = None
            else:
                # Try to use PyTorch if TensorFlow fails or model doesn't exist
                try:
                    import torch
                    from torchvision import models, transforms
                    from PIL import Image
                    
                    # Set device for PyTorch
                    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    print(f"🔄 Using PyTorch with device: {self.device}")
                    
                    # Load pre-trained ResNet model
                    self.general_model = models.resnet50(pretrained=True)
                    self.general_model.eval()
                    self.general_model.to(self.device)
                    
                    # Load labels
                    imagenet_labels_path = os.path.join(MODEL_DIR, 'imagenet_classes.txt')
                    if os.path.exists(imagenet_labels_path):
                        with open(imagenet_labels_path, "r") as f:
                            self.general_labels = f.read().splitlines()
                    else:
                        # Use simple list of class names
                        import requests
                        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
                        response = requests.get(url)
                        self.general_labels = response.text.strip().split('\n')
                    
                    print(f"✅ Loaded PyTorch ResNet model on {self.device}")
                    self.using_pytorch = True
                    
                except Exception as torch_error:
                    print(f"❌ Error loading PyTorch model: {str(torch_error)}")
                    self.model = None
                    self.labels = None
                    self.general_model = None
                    self.general_labels = None
                    self.using_pytorch = False
            
            # 2. Load Bird model and info
            print("🔄 Loading bird species classification model...")
            
            # Load the bird species model if available
            bird_model_path = os.path.join(MODEL_DIR, "bird_species_classifier.h5")
            if os.path.exists(bird_model_path):
                try:
                    import tensorflow as tf
                    self.bird_model = tf.keras.models.load_model(
                        bird_model_path,
                        custom_objects=custom_objects,
                        compile=False
                    )
                    self.bird_input_shape = get_model_input_shape(self.bird_model)
                    print(f"✅ Bird model loaded successfully from {bird_model_path}")
                    print(f"✅ Bird model input shape: {self.bird_input_shape}")
                    
                    # Load bird class labels
                    bird_labels_path = os.path.join(MODEL_DIR, "class_names.txt")
                    
                    if os.path.exists(bird_labels_path):
                        # Load the labels as list
                        with open(bird_labels_path, "r") as f:
                            self.bird_labels = [line.strip() for line in f.readlines()]
                        print(f"✅ Loaded bird species labels ({len(self.bird_labels)} classes)")
                    else:
                        print("⚠️ Bird species labels not found")
                        self.bird_labels = []
                        
                    # Try to load bird info from several possible locations
                    bird_info_locations = [
                        os.path.join(BIRDS_FOLDER_PATH, "bird_info_converted.json"),
                        os.path.join(MODEL_DIR, "bird_info_converted.json"),
                        os.path.join(MODEL_DIR, "bird_info.json")
                    ]
                    
                    self.bird_info = None
                    for bird_info_path in bird_info_locations:
                        if os.path.exists(bird_info_path):
                            with open(bird_info_path, "r") as f:
                                self.bird_info = json.load(f)
                            print(f"✅ Loaded bird info from {os.path.basename(bird_info_path)} ({len(self.bird_info)} entries)")
                            break
                    
                    if not self.bird_info:
                        print("⚠️ Bird info not found at any of the expected locations")
                        self.bird_info = {}
                
                except Exception as e:
                    print(f"❌ Error loading bird model: {str(e)}")
                    self.bird_model = None
                    self.bird_labels = None
                    self.bird_info = {}
            else:
                print("ℹ️ Bird species model not found, bird classification will not be available")
                self.bird_model = None
                self.bird_labels = None
                self.bird_info = {}
            
            # 3. Load Dog model and info
            print("🔄 Loading dog breed classifier...")
            dog_model_path = os.path.join(MODEL_DIR, 'dog_breed_classifier.h5')
            if os.path.exists(dog_model_path):
                try:
                    import tensorflow as tf
                    print(f"✅ Found dog model at {dog_model_path}, loading...")
                    self.dog_model = tf.keras.models.load_model(
                        dog_model_path,
                        custom_objects=custom_objects,
                        compile=False
                    )
                    self.dog_input_shape = get_model_input_shape(self.dog_model)
                    print(f"✅ Dog model loaded successfully from {dog_model_path}")
                    print(f"✅ Dog model input shape: {self.dog_input_shape}")
                except Exception as e:
                    self.dog_model = None
                    print(f"⚠️ Failed to load dog model: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️ Dog model file not found at {dog_model_path}")
                self.dog_model = None

            # Load dog labels
            dog_labels_path = os.path.join(MODEL_DIR, 'dog_class_names.txt')
            if os.path.exists(dog_labels_path):
                try:
                    with open(dog_labels_path, 'r') as f:
                        self.dog_labels = [line.strip() for line in f]
                    print(f"✅ Dog labels loaded successfully ({len(self.dog_labels)} classes)")
                    print(f"✅ First 5 dog labels: {self.dog_labels[:5]}")
                except Exception as e:
                    print(f"⚠️ Error loading dog labels: {str(e)}")
                    self.dog_labels = []
            else:
                print(f"⚠️ Dog labels file not found at {dog_labels_path}")
                self.dog_labels = []
            
            # Load dog breed info - Try multiple locations
            dog_info_paths = [
                os.path.join(MODEL_DIR, 'dog_breed_info.json'),
                os.path.join(PROJECT_ROOT, 'model', 'dog_breed_info.json'),
                os.path.join(PROJECT_ROOT, 'dog_breed_info.json')
            ]
            
            # First check if dog_breed_info exists in any location
            dog_info_path = None
            for path in dog_info_paths:
                print(f"🔍 Checking for dog breed info at: {path}")
                if os.path.exists(path):
                    dog_info_path = path
                    print(f"✅ Found dog breed info at {path}")
                    break
                else:
                    print(f"⚠️ No dog breed info at {path}")
            
            if dog_info_path:
                try:
                    print(f"🔄 Loading dog breed info from {dog_info_path}")
                    with open(dog_info_path, 'r', encoding='utf-8') as f:
                        self.dog_info = json.load(f)
                    print(f"✅ Dog breed info loaded successfully with {len(self.dog_info)} entries")
                    print(f"✅ Sample dog breeds in info: {list(self.dog_info.keys())[:5]}")
                    
                    # Check for mismatches between model labels and info file
                    if self.dog_labels:
                        known_breed_count = 0
                        unknown_breeds = []
                        
                        for breed in self.dog_labels:
                            # Try different forms of the breed name
                            standard_breed_name = ' '.join(w.capitalize() for w in breed.split())
                            found = False
                            
                            if breed in self.dog_info or standard_breed_name in self.dog_info:
                                known_breed_count += 1
                                found = True
                            else:
                                # Try case-insensitive matching
                                for info_breed in self.dog_info.keys():
                                    if breed.lower() == info_breed.lower() or \
                                       breed.lower().replace(' ', '') == info_breed.lower().replace(' ', ''):
                                        known_breed_count += 1
                                        found = True
                                        break
                            
                            if not found:
                                unknown_breeds.append(breed)
                        
                        if unknown_breeds:
                            print(f"⚠️ Found {len(unknown_breeds)} breeds in model without matching info")
                            print(f"⚠️ Examples: {unknown_breeds[:5]}")
                        
                        print(f"✅ {known_breed_count} out of {len(self.dog_labels)} breeds have matching info")
                except Exception as e:
                    print(f"⚠️ Error loading dog breed info: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    self.dog_info = {}
            else:
                print("⚠️ No dog breed info file found in any location, creating empty dictionary")
                self.dog_info = {}
            
            # 4. Load Flower model and info
            print("🔄 Loading flower classifier...")
            flower_model_path = os.path.join(MODEL_DIR, 'flower_classifier.h5')
            if os.path.exists(flower_model_path):
                try:
                    # Attempt to load with custom objects
                    import tensorflow as tf
                    self.flower_model = tf.keras.models.load_model(
                        flower_model_path,
                        custom_objects=custom_objects,
                        compile=False
                    )
                    self.flower_input_shape = get_model_input_shape(self.flower_model)
                    print(f"✅ Flower model loaded successfully from {flower_model_path}")
                    print(f"✅ Flower model input shape: {self.flower_input_shape}")
                except Exception as e:
                    # Try loading without custom objects
                    try:
                        print(f"⚠️ First attempt failed, trying alternative loading method...")
                        import tensorflow as tf
                        self.flower_model = tf.keras.models.load_model(flower_model_path, compile=False)
                        self.flower_input_shape = get_model_input_shape(self.flower_model)
                        print(f"✅ Flower model loaded successfully on second attempt")
                    except Exception as e2:
                        self.flower_model = None
                        print(f"⚠️ Failed to load flower model: {str(e2)}")
            else:
                print(f"⚠️ Flower model file not found at {flower_model_path}")
                self.flower_model = None

            # Load flower labels
            flower_labels_path = os.path.join(MODEL_DIR, 'classnames.txt')
            if os.path.exists(flower_labels_path):
                with open(flower_labels_path, 'r', encoding='utf-8') as f:
                    self.flower_labels = [line.strip() for line in f]
                print(f"✅ Flower labels loaded successfully ({len(self.flower_labels)} classes)")
            else:
                print(f"⚠️ Flower labels file not found at {flower_labels_path}")
                self.flower_labels = []

            # Load flower information
            flower_info_path = os.path.join(MODEL_DIR, 'flower_info_varied.json')
            if os.path.exists(flower_info_path):
                with open(flower_info_path, 'r', encoding='utf-8') as f:
                    self.flower_info = json.load(f)
                print(f"✅ Flower information loaded successfully from {flower_info_path}")
                print(f"✅ Found information for {len(self.flower_info)} flower species")
                # Print sample of keys for verification
                sample_keys = list(self.flower_info.keys())[:3]
                print(f"✅ Sample flower keys: {sample_keys}")
            else:
                print(f"⚠️ Flower info file not found at {flower_info_path}")
                self.flower_info = {}
            
            # 5. Load Plant model and info
            print("🔄 Loading plant classifier...")
            plant_model_path = os.path.join(MODEL_DIR, 'plant_classifier.h5')
            if os.path.exists(plant_model_path):
                try:
                    import tensorflow as tf
                    self.plant_model = tf.keras.models.load_model(
                        plant_model_path,
                        custom_objects=custom_objects,
                        compile=False
                    )
                    self.plant_input_shape = get_model_input_shape(self.plant_model)
                    print(f"✅ Plant model loaded successfully from {plant_model_path}")
                    print(f"✅ Plant model input shape: {self.plant_input_shape}")
                except Exception as e:
                    self.plant_model = None
                    print(f"⚠️ Failed to load plant model: {str(e)}")
            else:
                print(f"⚠️ Plant model file not found at {plant_model_path}")
                self.plant_model = None
                
            # Load plant labels - try multiple sources
            self.plant_labels = []
            
            # First try class_labels.json
            plant_labels_path = os.path.join(MODEL_DIR, 'class_labels.json')
            if os.path.exists(plant_labels_path):
                try:
                    with open(plant_labels_path, 'r', encoding='utf-8') as f:
                        plant_labels_dict = json.load(f)
                        # Convert dictionary to list ensuring correct order by index
                        max_index = max(plant_labels_dict.values())
                        self.plant_labels = [""] * (max_index + 1)
                        for label, index in plant_labels_dict.items():
                            self.plant_labels[index] = label
                    print(f"✅ Plant labels loaded successfully from class_labels.json ({len(self.plant_labels)} classes)")
                    # Normalize labels - replace underscores with spaces and capitalize
                    self.plant_labels = [label.replace('_', ' ').title() for label in self.plant_labels]
                except Exception as e:
                    print(f"⚠️ Error loading plant labels from {plant_labels_path}: {str(e)}")
                    self.plant_labels = []
            else:
                print(f"⚠️ Plant labels file not found at {plant_labels_path}")
                self.plant_labels = []
             
            # If that failed, try class_names.txt
            if not self.plant_labels:
                plant_labels_path = os.path.join(MODEL_DIR, 'class_names.txt')
                if os.path.exists(plant_labels_path):
                    try:
                        with open(plant_labels_path, 'r') as f:
                            self.plant_labels = [line.strip() for line in f]
                        print(f"✅ Plant labels loaded successfully from class_names.txt ({len(self.plant_labels)} classes)")
                    except Exception as e:
                        print(f"⚠️ Error loading plant labels from {plant_labels_path}: {str(e)}")
             
            # If still no labels, extract them from plant_info.json as a fallback
            if not self.plant_labels:
                plant_info_path = os.path.join(MODEL_DIR, 'plant_info.json')
                if os.path.exists(plant_info_path):
                    try:
                        with open(plant_info_path, 'r', encoding='utf-8') as f:
                            plant_info = json.load(f)
                            self.plant_labels = list(plant_info.keys())
                        print(f"✅ Plant labels extracted from plant_info.json as fallback ({len(self.plant_labels)} classes)")
                    except Exception as e:
                        print(f"⚠️ Error extracting plant labels from plant_info.json: {str(e)}")
                        self.plant_labels = []
            
            # Load plant info
            plant_info_path = os.path.join(MODEL_DIR, 'plant_info.json')
            if os.path.exists(plant_info_path):
                with open(plant_info_path, 'r', encoding='utf-8') as f:
                    self.plant_info = json.load(f)
                print(f"✅ Plant information loaded successfully ({len(self.plant_info)} entries)")
            else:
                print(f"⚠️ Plant info file not found at {plant_info_path}")
                self.plant_info = {}
            
            # 6. Load Insect model and info
            print("🔄 Loading insect classifier...")
            insect_model_path = os.path.join(MODEL_DIR, 'insect_model.h5')
            if not os.path.exists(insect_model_path):
                insect_model_path = os.path.join(MODEL_DIR, 'insect_model11.h5')
                
            if os.path.exists(insect_model_path):
                try:
                    import tensorflow as tf
                    self.insect_model = tf.keras.models.load_model(
                        insect_model_path,
                        custom_objects=custom_objects,
                        compile=False
                    )
                    self.insect_input_shape = get_model_input_shape(self.insect_model)
                    print(f"✅ Insect model loaded successfully from {insect_model_path}")
                    print(f"✅ Insect model input shape: {self.insect_input_shape}")
                except Exception as e:
                    self.insect_model = None
                    print(f"⚠️ Failed to load insect model: {str(e)}")
            else:
                print(f"⚠️ Insect model file not found at {insect_model_path}")
                self.insect_model = None
                
            # Load insect labels
            insect_labels_path = os.path.join(MODEL_DIR, 'insect_labels.json')
            if os.path.exists(insect_labels_path):
                with open(insect_labels_path, 'r', encoding='utf-8') as f:
                    self.insect_labels = json.load(f)
                print(f"✅ Insect labels loaded successfully ({len(self.insect_labels)} classes)")
            else:
                print(f"⚠️ Insect labels file not found at {insect_labels_path}")
                self.insect_labels = {}
                
            # Load insect info
            insect_info_path = os.path.join(MODEL_DIR, 'insect_info.json')
            if os.path.exists(insect_info_path):
                with open(insect_info_path, 'r', encoding='utf-8') as f:
                    self.insect_info = json.load(f)
                print(f"✅ Insect information loaded successfully ({len(self.insect_info)} entries)")
            else:
                print(f"⚠️ Insect info file not found at {insect_info_path}")
                self.insect_info = {}
            
            # 7. Ensure cat info is available
            print("🔄 Checking cat information dictionary...")
            if not hasattr(self, 'cat_info') or not self.cat_info:
                print("✅ Loading cat information dictionary")
                # The cat_info dictionary is already defined in the class
            
            # Report on what models were loaded successfully
            print("\n📋 MODEL LOADING SUMMARY:")
            print(f"🦜 Bird model: {'Available' if self.bird_model else 'Not available'}")
            print(f"🐕 Dog model: {'Available' if self.dog_model else 'Not available'}")
            print(f"🌸 Flower model: {'Available' if self.flower_model else 'Not available'}")
            print(f"🌿 Plant model: {'Available' if self.plant_model else 'Not available'}")
            print(f"🐛 Insect model: {'Available' if self.insect_model else 'Not available'}")
            print(f"🖼️ General model: {'Available' if self.general_model or self.model else 'Not available'}")
            
            # Check if we have at least one model loaded
            if self.model or self.general_model or self.bird_model or self.dog_model or self.flower_model or self.plant_model or self.insect_model:
                self.models_loaded = True
                print("✅ Models loaded successfully")
            else:
                print("⚠️ No models were loaded successfully")
                self.models_loaded = False
            
            print(f"🔄 LOADING MODELS COMPLETE\n\n")
            
            return self.models_loaded
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            import traceback
            traceback.print_exc()
            self.models_loaded = False
            return False

    def predict_bird(self, image_path: str) -> Optional[Dict[str, Union[str, float]]]:
        """
        Predict bird species from an image.
        
        Args:
            image_path: Path to the image file to classify
            
        Returns:
            Dictionary with prediction results or None if prediction failed
        """
        if self.bird_model is None or not self.bird_labels:
            print("⚠️ Bird species model or labels not available")
            return None
            
        try:
            print(f"\n\n🦜🦜🦜 BIRD PREDICTION STARTING 🦜🦜🦜")
            print(f"🦜 Starting bird prediction for image: {image_path}")
            
            # First validate the image exists
            if not os.path.exists(image_path):
                print(f"❌ Error: Image file not found: {image_path}")
                return {
                    'label': 'Image not found',
                    'confidence': 0.0,
                    'description': f'The image file could not be found at {image_path}'
                }
                
            # Use enhanced preprocessing for birds
            print("🔄 Preprocessing bird image")
            img = Image.open(image_path).convert('RGB')
            
            # Resize and preprocess
            img = img.resize(self.bird_input_shape)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Get predictions from the model
            print("🔄 Running bird species prediction")
            predictions = self.bird_model.predict(img_array)
            
            # Get the top 3 results for better analysis
            top_indices = np.argsort(-predictions[0])[:3]
            top_predictions = []
            
            print("🔍 Top 3 bird species predictions:")
            for i, idx in enumerate(top_indices):
                if idx < len(self.bird_labels):
                    label = self.bird_labels[idx]
                    confidence = float(predictions[0][idx])
                    top_predictions.append((label, confidence))
                    print(f"   {i+1}. {label} ({confidence:.4f} = {confidence*100:.2f}%)")
            
            # If no valid predictions, return error
            if not top_predictions:
                print("❌ No valid bird species predictions found")
                return {
                    'label': 'No bird species identified',
                    'confidence': 0.0,
                    'description': 'The model could not identify any bird species in this image. Try a clearer image of a bird.'
                }
            
            # Use the top prediction
            bird_name, confidence = top_predictions[0]

            # Special case handling for specific birds
            if bird_name == "Scarlet_Tanager" or bird_name.lower() == "scarlet_tanager" or bird_name.lower() == "scarlet tanager":
                print("🎯 Direct match for Scarlet Tanager detected")
                # Return the specific data directly without going through matching
                result = {
                    'label': "Scarlet Tanager",
                    'confidence': confidence * 100,
                    'scientific_name': "Piranga olivacea",
                    'habitat': "Mature deciduous forests with tall trees, especially oak forests",
                    'diet': "Mainly insects, especially bees, wasps, beetles, and caterpillars; also berries and small fruits",
                    'lifespan': "Up to 10 years in the wild",
                    'weight': "23-38 g (0.8-1.3 oz)",
                    'height': "16-19 cm (6.3-7.5 in)",
                    'wingspan': "25-30 cm (9.8-11.8 in)",
                    'description': "Male in breeding plumage has brilliant scarlet-red body with jet-black wings and tail. Females and fall males are olive-green above, yellowish below, with darker olive wings. Has a stout, pale bill and relatively short tail.",
                    'interesting_fact': "Despite their bright coloration, Scarlet Tanagers can be difficult to spot as they tend to stay high in the forest canopy. Their harsh, burry song sounds like a robin with a sore throat."
                }
                print(f"📊 Returning specialized Scarlet Tanager prediction: ({confidence*100:.2f}%)")
                print(f"🦜🦜🦜 BIRD PREDICTION COMPLETE 🦜🦜🦜\n\n")
                return result

            # If confidence is too low, provide feedback
            if confidence < 0.3:  # less than 30% confidence
                print(f"⚠️ Low confidence prediction: {bird_name} ({confidence*100:.2f}%)")
            
            # Prepare search keys
            search_keys = [
                bird_name.lower(),
                bird_name.lower().replace(' ', '_'),
                ' '.join(bird_name.lower().split(' ')[:-1]) if ' ' in bird_name else bird_name.lower()  # Remove last word (might be 'bird')
            ]
            
            # Handle common name variations
            if "common" in bird_name.lower():
                search_keys.append(bird_name.lower().replace("common ", ""))
            
            if "northern" in bird_name.lower():
                search_keys.append(bird_name.lower().replace("northern ", ""))
                
            if "eastern" in bird_name.lower():
                search_keys.append(bird_name.lower().replace("eastern ", ""))
                
            if "western" in bird_name.lower():
                search_keys.append(bird_name.lower().replace("western ", ""))
                
            if "american" in bird_name.lower():
                search_keys.append(bird_name.lower().replace("american ", ""))
                
            if "european" in bird_name.lower():
                search_keys.append(bird_name.lower().replace("european ", ""))
                
            # For species with color in name, add version without the color
            colors = ["red", "blue", "green", "yellow", "black", "white", "brown", "gray", "grey"]
            for color in colors:
                if color in bird_name.lower():
                    search_keys.append(bird_name.lower().replace(f"{color} ", ""))
                    search_keys.append(bird_name.lower().replace(f"{color}-", ""))
            
            # Add specific handling for birds with underscores
            if "_" in bird_name:
                # For names like Scarlet_Tanager, try both "Scarlet Tanager" and "Scarlet" + "Tanager" separately
                parts = bird_name.split("_")
                for part in parts:
                    search_keys.append(part.lower())
                
                # Also add with spaces instead of underscores
                search_keys.append(bird_name.lower().replace("_", " "))

            # Try variations without hyphens and underscores
            clean_name = bird_name.lower().replace("_", " ").replace("-", " ")
            search_keys.append(clean_name)

            # Try the name without any special characters or spaces (e.g., "scarlet tanager" -> "scarlettanager")
            search_keys.append(bird_name.lower().replace("_", "").replace("-", "").replace(" ", ""))
            
            print(f"🔍 Trying to match with search keys: {search_keys}")
            
            # Match with bird info 
            matched_info = None
            matched_key = None
            
            # First try direct matches with our search keys
            for key in search_keys:
                if key in self.bird_info:
                    matched_info = self.bird_info[key]
                    matched_key = key
                    print(f"✅ Found exact bird info match for: {key}")
                    break
            
            # If no direct match, try fuzzy match
            if not matched_info:
                print(f"⚠️ No exact match found for '{bird_name}' in bird info")
                best_match_score = 0
                
                for bird_key in self.bird_info.keys():
                    # Try various matching strategies
                    for search_key in search_keys:
                        # Check if search_key is in bird_key or vice versa
                        if search_key in bird_key or bird_key in search_key:
                            # Calculate match score based on length of common substring
                            match_score = len(search_key) / max(len(search_key), len(bird_key)) if search_key else 0
                            
                            if match_score > best_match_score:
                                best_match_score = match_score
                                matched_info = self.bird_info[bird_key]
                                matched_key = bird_key
                
                if matched_key:
                    print(f"✅ Found fuzzy match: '{matched_key}' for '{bird_name}' (score: {best_match_score:.2f})")
            
            # If still no match, use generic info for bird species or create basic info
            if not matched_info:
                print(f"⚠️ No bird info match found after all attempts, creating generic info")
                # Create basic info with what we know
                bird_name_display = bird_name.replace('_', ' ')
                matched_info = {
                    "Scientific Name": f"{bird_name.title().replace('_', ' ')}",
                    "Habitat": "Forests, wetlands, grasslands, and urban environments",
                    "Diet": "Seeds, insects, fruits, nectar, and small vertebrates depending on the species",
                    "Lifespan": "4-8 years in the wild, up to 15 years in captivity",
                    "Weight": "20-150 grams depending on the species",
                    "Height": "15-25 cm (6-10 inches)",
                    "Wingspan": "25-40 cm (10-16 inches)",
                    "Description": f"The {bird_name_display} is characterized by its distinctive plumage, beak shape, and behavior adapted to its ecological niche.",
                    "Interesting Fact": f"The {bird_name_display} is known for its unique vocalizations and nesting behaviors that have evolved over thousands of years."
                }
                matched_key = bird_name.lower()
            
            # Check for missing fields and provide defaults or check for "Scientificus birdus" placeholder
            required_fields = ["Scientific Name", "Habitat", "Diet", "Lifespan", "Weight", 
                              "Height", "Wingspan", "Description", "Interesting Fact"]
            
            for field in required_fields:
                if (field not in matched_info or 
                    not matched_info[field] or 
                    matched_info[field] == "Unknown" or 
                    "Varies by" in matched_info.get(field, "") or
                    "Scientificus birdus" in matched_info.get(field, "")):
                    
                    print(f"⚠️ Missing, generic or placeholder {field} for {bird_name}, providing default value")
                    bird_name_display = bird_name.replace('_', ' ')
                    
                    if field == "Scientific Name":
                        # Improve scientific name formatting to be more realistic
                        if "Scientificus birdus" in matched_info.get(field, ""):
                            genus = bird_name.split("_")[0].capitalize() if "_" in bird_name else bird_name.split(" ")[0].capitalize()
                            species = bird_name.split("_")[-1].lower() if "_" in bird_name else bird_name.split(" ")[-1].lower()
                            matched_info[field] = f"{genus} {species}"
                        else:
                            matched_info[field] = f"{bird_name.title().replace('_', ' ')}"
                    elif field == "Habitat":
                        matched_info[field] = "Forests, wetlands, grasslands, and urban environments"
                    elif field == "Diet":
                        matched_info[field] = "Seeds, insects, fruits, nectar, and small vertebrates depending on the species"
                    elif field == "Lifespan":
                        matched_info[field] = "4-8 years in the wild, up to 15 years in captivity"
                    elif field == "Weight":
                        if "_" in bird_name or " " in bird_name:
                            # More specific for known birds
                            size_class = "small"
                            if any(term in bird_name.lower() for term in ["eagle", "hawk", "owl", "falcon"]):
                                size_class = "large"
                                matched_info[field] = "800-2000 grams"
                            elif any(term in bird_name.lower() for term in ["jay", "robin", "tanager", "warbler"]):
                                size_class = "small"
                                matched_info[field] = "20-80 grams"
                            elif any(term in bird_name.lower() for term in ["finch", "sparrow", "bunting", "titmouse"]):
                                size_class = "very small"
                                matched_info[field] = "10-35 grams"
                            else:
                                matched_info[field] = "30-150 grams" # Medium
                        else:
                            matched_info[field] = "20-150 grams depending on the species"
                    elif field == "Height":
                        if "_" in bird_name or " " in bird_name:
                            # More specific for known birds
                            if any(term in bird_name.lower() for term in ["eagle", "hawk", "owl", "falcon"]):
                                matched_info[field] = "45-90 cm (18-35 inches)"
                            elif any(term in bird_name.lower() for term in ["jay", "robin", "tanager", "warbler"]):
                                matched_info[field] = "15-25 cm (6-10 inches)"
                            elif any(term in bird_name.lower() for term in ["finch", "sparrow", "bunting", "titmouse"]):
                                matched_info[field] = "10-18 cm (4-7 inches)"
                            else:
                                matched_info[field] = "20-40 cm (8-16 inches)" # Medium
                        else:
                            matched_info[field] = "15-25 cm (6-10 inches)"
                    elif field == "Wingspan":
                        if "_" in bird_name or " " in bird_name:
                            # More specific for known birds
                            if any(term in bird_name.lower() for term in ["eagle", "hawk", "owl", "falcon"]):
                                matched_info[field] = "120-220 cm (47-87 inches)"
                            elif any(term in bird_name.lower() for term in ["jay", "robin", "tanager", "warbler"]):
                                matched_info[field] = "20-35 cm (8-14 inches)"
                            elif any(term in bird_name.lower() for term in ["finch", "sparrow", "bunting", "titmouse"]):
                                matched_info[field] = "15-25 cm (6-10 inches)"
                            else:
                                matched_info[field] = "30-60 cm (12-24 inches)" # Medium
                        else:
                            matched_info[field] = "25-40 cm (10-16 inches)"
                    elif field == "Description":
                        # More detailed description based on bird name
                        color_terms = {
                            "red": "reddish", "blue": "bluish", "green": "greenish", 
                            "yellow": "yellowish", "black": "black", "white": "white",
                            "brown": "brownish", "gray": "grayish", "grey": "greyish",
                            "scarlet": "bright red", "indigo": "deep blue", "rusty": "rusty-colored"
                        }
                        
                        bird_colors = []
                        for color, description in color_terms.items():
                            if color in bird_name.lower():
                                bird_colors.append(description)
                        
                        color_desc = ""
                        if bird_colors:
                            color_desc = f" with {', '.join(bird_colors)} plumage"
                        
                        bird_parts = bird_name.replace("_", " ").split()
                        part_desc = ""
                        for part in ["throated", "headed", "bellied", "winged", "tailed", "backed", "breasted"]:
                            if any(part in word for word in bird_parts):
                                part_desc = f" and distinctive markings"
                                break
                        
                        matched_info[field] = f"The {bird_name_display} is a bird species{color_desc}{part_desc}. They are known for their distinctive appearance and behaviors specific to their habitat."
                    elif field == "Interesting Fact":
                        matched_info[field] = f"The {bird_name_display} has unique adaptations that help it thrive in its natural environment."
            
            # Build the result
            result = {
                'label': bird_name.replace('_', ' '),  # Replace underscores with spaces for display
                'confidence': confidence * 100,  # Convert to percentage for display
                'scientific_name': matched_info.get("Scientific Name", "Unknown"),
                'habitat': matched_info.get("Habitat", "Various habitats"),
                'diet': matched_info.get("Diet", "Various foods"),
                'lifespan': matched_info.get("Lifespan", "Unknown"),
                'weight': matched_info.get("Weight", "Unknown"),
                'height': matched_info.get("Height", "Unknown"),
                'wingspan': matched_info.get("Wingspan", "Unknown"),
                'description': matched_info.get("Description", "No description available"),
                'interesting_fact': matched_info.get("Interesting Fact", "No additional information available")
            }
            
            print(f"📊 Returning bird species prediction: {bird_name} ({confidence*100:.2f}%)")
            print(f"🦜🦜🦜 BIRD PREDICTION COMPLETE 🦜🦜🦜\n\n")
            return result
            
        except Exception as img_error:
            print(f"❌ Error during bird image processing: {str(img_error)}")
            import traceback
            traceback.print_exc()
            return {
                'label': 'Error during analysis',
                'confidence': 0.0,
                'description': f'An error occurred during bird species analysis: {str(img_error)}. Please try again with a different image.'
            }

    def predict_general(self, image_path: str) -> Optional[Dict[str, Union[str, float]]]:
        """
        Predict general image category using ResNet model.
        Enriches result with additional information if available.
        """
        try:
            print(f"\n\n🖼️🖼️🖼️ GENERAL PREDICTION STARTING 🖼️🖼️🖼️")
            print(f"🖼️ Starting general prediction for {image_path}")
            
            # Load the general model if not loaded
            if not self.general_model:
                print("🔄 Loading general classification model...")
                try:
                    # Try to load ResNet model
                    self.general_model = models.resnet50(pretrained=True)
                    self.general_model.eval()
                    
                    # Load labels
                    imagenet_labels_path = os.path.join(MODEL_DIR, 'imagenet_classes.txt')
                    if os.path.exists(imagenet_labels_path):
                        with open(imagenet_labels_path, 'r') as f:
                            self.general_labels = f.read().splitlines()
                    else:
                        # Use simple list of class names
                        import requests
                        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
                        response = requests.get(url)
                        self.general_labels = response.text.strip().split('\n')
                    
                    print(f"✅ General model and labels loaded successfully ({len(self.general_labels)} classes)")
                except Exception as e:
                    print(f"❌ Failed to load general model: {str(e)}")
                    return None
            
            if not self.general_model or not self.general_labels:
                print("❌ Error: General classification model not loaded")
                return None
            
            # Validate image exists
            if not os.path.exists(image_path):
                print(f"❌ Error: Image file not found: {image_path}")
                return None
                
            # Preprocess image for ResNet
            try:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                
                img = Image.open(image_path).convert('RGB')
                img_t = transform(img)
                batch_t = torch.unsqueeze(img_t, 0)
                
                # Make prediction
                with torch.no_grad():
                    output = self.general_model(batch_t)
                
                # Get top 3 predictions for better results
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                top3_prob, top3_indices = torch.topk(probabilities, 3)
                
                # Convert to Python types
                top_predictions = [(self.general_labels[idx], float(prob)) 
                                  for prob, idx in zip(top3_prob, top3_indices)]
                
                # Print top predictions
                print("🔍 Top 3 general predictions:")
                for i, (label, confidence) in enumerate(top_predictions):
                    print(f"   {i+1}. {label} ({confidence:.4f})")
                
                # Use top prediction
                label, confidence = top_predictions[0]
                
                # Check for cat-related labels
                cat_match = None
                is_cat = False
                
                # Create search keys for cat matches
                cat_search_terms = [label.lower()]
                if 'cat' in label.lower():
                    is_cat = True
                    cat_search_terms.extend(['cat', label.lower().split(' ')[0] + ' cat'])
                    
                # Special mappings for common ImageNet cat classes
                cat_mapping = {
                    'egyptian_cat': 'egyptian cat',
                    'persian_cat': 'persian cat',
                    'siamese_cat': 'siamese cat',
                    'tabby': 'tabby cat',
                    'tabby_cat': 'tabby cat',
                    'tiger_cat': 'tabby cat'
                }
                
                # Apply any special mapping
                for key, value in cat_mapping.items():
                    if key in label.lower():
                        cat_search_terms.append(value)
                
                # Try to find a match in cat_info
                if is_cat:
                    print(f"🐱 Detected cat in the image: {label}")
                    for cat_key in self.cat_info:
                        for search_term in cat_search_terms:
                            if search_term in cat_key or cat_key in search_term:
                                cat_match = self.cat_info[cat_key]
                                print(f"✅ Found cat match: '{cat_key}' for '{label}'")
                                break
                        if cat_match:
                            break
                    
                    # If still no match but it is a cat, use generic cat info
                    if not cat_match and is_cat:
                        cat_match = self.cat_info.get('cat', {})
                        print(f"✅ Using generic cat info for '{label}'")
                
                # Check if it's a bird
                bird_info = None
                if not cat_match and self.bird_info:
                    for bird_key in self.bird_info:
                        if label.lower() in bird_key.lower() or bird_key.lower() in label.lower():
                            bird_info = self.bird_info[bird_key]
                            print(f"✅ Found bird match: '{bird_key}' for '{label}'")
                            break
                
                # Create the result with the label, confidence and other context
                result = {
                    'label': label,
                    'confidence': confidence,
                    'category': 'general'
                }
                
                # Add cat information if available
                if cat_match:
                    result.update({
                        'scientific_name': cat_match.get('scientific_name', 'Felis catus'),
                        'description': cat_match.get('description', 'No description available'),
                        'origin': cat_match.get('origin', 'Unknown'),
                        'temperament': cat_match.get('temperament', 'Unknown'),
                        'lifespan': cat_match.get('lifespan', 'Unknown'),
                        'weight': cat_match.get('weight', 'Unknown')
                    })
                    
                # Add bird information if available and not a cat
                elif bird_info:
                    result.update({
                        'scientific_name': bird_info.get('Scientific Name', 'Unknown'),
                        'description': bird_info.get('Description', 'No description available')
                    })
                
                print(f"📊 General prediction result: {label} ({confidence*100:.2f}%)")
                print(f"🖼️🖼️🖼️ GENERAL PREDICTION COMPLETE 🖼️🖼️🖼️\n\n")
                return result
                
            except Exception as img_error:
                print(f"❌ Error processing image: {str(img_error)}")
                import traceback
                traceback.print_exc()
                return None
                
        except Exception as e:
            print(f"❌ General prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def predict_dog(self, image_path: str) -> Optional[Dict[str, Union[str, float]]]:
        """
        Predict dog breed from an image.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Dictionary with prediction results or None if prediction failed
        """
        print(f"\n\n🐕🐕🐕 DOG BREED PREDICTION STARTING 🐕🐕🐕")
        print(f"🐕 Starting dog breed prediction for image: {image_path}")
        
        # Check if specialized dog model is available
        use_general_model = False
        if self.dog_model is None or not self.dog_labels:
            print("⚠️ Specialized dog breed model or labels not available, will use general model as fallback")
            use_general_model = True
            
            # Check if general model is available
            if (self.model is None and self.general_model is None):
                print("❌ Neither specialized dog model nor general model is available")
                return {
                    'label': 'Dog Breed Analyzer Not Available',
                    'confidence': 0.0,
                    'description': 'The dog breed classifier is not available. Please check if any model file exists.'
                }
        
        # Check if TensorFlow is available (needed for both specialized and general model)
        if not is_tensorflow_available():
            print("⚠️ TensorFlow is not available. Cannot predict dog breed.")
            return {
                "label": "Dog Breed Analyzer Not Available",
                "confidence": 0.0,
                "description": "Dog breed classification requires TensorFlow which is not installed."
            }
        
        try:
            # First validate the image exists
            if not os.path.exists(image_path):
                print(f"❌ Error: Image file not found: {image_path}")
                return {
                    'label': 'Image not found',
                    'confidence': 0.0,
                    'description': f'The image file could not be found at {image_path}'
                }
            
            # Double check image can be opened
            try:
                with Image.open(image_path) as img:
                    print(f"✅ Successfully opened image with size: {img.size}")
                    if img.size[0] < 50 or img.size[1] < 50:
                        print(f"⚠️ Warning: Image is very small: {img.size}")
            except Exception as img_err:
                print(f"❌ Error opening image: {str(img_err)}")
                return {
                    'label': 'Invalid Image',
                    'confidence': 0.0,
                    'description': f'The image file could not be processed: {str(img_err)}'
                }
            
            # If using specialized dog model
            if not use_general_model:
                try:
                    # Preprocess the image for the dog model
                    img_array = self._preprocess_keras(image_path, self.dog_input_shape)
                    print(f"✅ Successfully preprocessed image, shape: {img_array.shape}")
                    
                    # Get predictions from the model
                    print("🔄 Running specialized dog breed prediction")
                    predictions = self.dog_model.predict(img_array, verbose=0)
                    print(f"✅ Successfully got predictions with shape: {predictions.shape}")
                    
                    # Get the top 3 results for better analysis
                    top_indices = np.argsort(-predictions[0])[:3]
                    top_predictions = []
                    
                    print("🔍 Top 3 dog breed predictions:")
                    for i, idx in enumerate(top_indices):
                        if idx < len(self.dog_labels):
                            label = self.dog_labels[idx]
                            confidence = float(predictions[0][idx])
                            top_predictions.append((label, confidence))
                            print(f"   {i+1}. {label} ({confidence:.4f} = {confidence*100:.2f}%)")
                    
                    # If no valid predictions, fall back to general model
                    if not top_predictions:
                        print("⚠️ No valid dog breed predictions found with specialized model, falling back to general model")
                        use_general_model = True
                    else:
                        # Continue with specialized model results
                        breed_name, confidence = top_predictions[0]
                except Exception as e:
                    print(f"❌ Error using specialized dog model: {str(e)}")
                    print("⚠️ Falling back to general model")
                    use_general_model = True
            
            # If using general model (either as primary or fallback)
            if use_general_model:
                print("🔄 Using general model for dog breed prediction")
                general_result = self.predict_general(image_path)
                
                if not general_result:
                    print("❌ General model prediction failed")
                    return {
                        'label': 'Prediction Failed',
                        'confidence': 0.0,
                        'description': 'The general model failed to process the image. Please try a different image.'
                    }
                
                # Filter results to only include dog-related classes
                dog_keywords = ['dog', 'terrier', 'spaniel', 'retriever', 'shepherd', 'hound', 'poodle', 'bulldog', 'mastiff', 'collie']
                
                label = general_result.get('label', '').lower()
                is_dog = any(keyword in label for keyword in dog_keywords)
                
                if not is_dog:
                    print(f"⚠️ General model did not identify a dog: {label}")
                    return {
                        'label': 'Not a Dog',
                        'confidence': general_result.get('confidence', 0.0),
                        'description': 'The image does not appear to contain a dog. Please try an image with a clear view of a dog.'
                    }
                
                print(f"✅ General model identified a dog: {general_result.get('label')}")
                breed_name = general_result.get('label', 'Unknown Dog')
                confidence = general_result.get('confidence', 0.0) / 100.0  # Convert percentage back to ratio
            
            # Process the breed information (works with both specialized and general model results)
            # Get breed info from the cached dog_info
            breed_info = None
            
            if self.dog_info:
                print(f"✅ Dog info dictionary available with {len(self.dog_info)} entries")
                try:
                    # Try to find the breed in the info dictionary using improved matching algorithm
                    
                    # 1. Try exact match first
                    if breed_name in self.dog_info:
                        breed_info = self.dog_info[breed_name]
                        print(f"✅ Found exact match for '{breed_name}' in dog_info")
                    else:
                        # 2. Try standard case matching
                        standard_breed_name = ' '.join(w.capitalize() for w in breed_name.split())
                        if standard_breed_name in self.dog_info:
                            breed_info = self.dog_info[standard_breed_name]
                            breed_name = standard_breed_name  # Use the name from the dictionary
                            print(f"✅ Found standard case match: '{standard_breed_name}' for '{breed_name}'")
                        else:
                            # 3. Try case-insensitive match
                            for info_breed in self.dog_info.keys():
                                if breed_name.lower() == info_breed.lower() or \
                                breed_name.lower().replace(' ', '') == info_breed.lower().replace(' ', ''):
                                    breed_info = self.dog_info[info_breed]
                                    breed_name = info_breed  # Use the name from the dictionary
                                    print(f"✅ Found case-insensitive match: '{info_breed}' for '{breed_name}'")
                                    break
                            
                            # 4. If still no match, try partial match
                            if not breed_info:
                                for info_breed in self.dog_info.keys():
                                    if breed_name.lower() in info_breed.lower() or info_breed.lower() in breed_name.lower():
                                        breed_info = self.dog_info[info_breed]
                                        breed_name = info_breed  # Use the name from the dictionary
                                        print(f"✅ Found partial match: '{info_breed}' for '{breed_name}'")
                                        break
                                        
                                # 5. Try removing common suffix/prefix words
                                if not breed_info:
                                    base_breed_name = breed_name.lower()
                                    for suffix in ['dog', 'hound', 'terrier', 'spaniel', 'retriever', 'shepherd']:
                                        if base_breed_name.endswith(suffix):
                                            base_breed_name = base_breed_name[:-len(suffix)].strip()
                                    
                                    for info_breed in self.dog_info.keys():
                                        base_info_breed = info_breed.lower()
                                        for suffix in ['dog', 'hound', 'terrier', 'spaniel', 'retriever', 'shepherd']:
                                            if base_info_breed.endswith(suffix):
                                                base_info_breed = base_info_breed[:-len(suffix)].strip()
                                        
                                        if base_breed_name == base_info_breed or base_breed_name in base_info_breed or base_info_breed in base_breed_name:
                                            breed_info = self.dog_info[info_breed]
                                            breed_name = info_breed  # Use the name from the dictionary
                                            print(f"✅ Found base name match: '{info_breed}' for '{breed_name}'")
                                            break
                except Exception as e:
                    print(f"⚠️ Error finding breed info: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("⚠️ Dog info dictionary is empty or not available")
            
            # Create generic info if no match found
            if not breed_info:
                print(f"⚠️ No breed info found for '{breed_name}', using generic info")
                breed_info = {
                    "scientific_name": "Canis lupus familiaris",
                    "height": "Varies by individual",
                    "weight": "Varies by individual",
                    "life_span": "10-15 years",
                    "temperament": "Varies by individual",
                    "origin": "Selective breeding",
                    "group": "Unknown",
                    "coat": "Varies",
                    "colors": "Varies",
                    "description": f"This appears to be a {breed_name} dog. Unfortunately, detailed information about this specific breed is not available in our database."
                }
            
            # Ensure all required fields are present
            required_fields = ["scientific_name", "height", "weight", "life_span", "temperament", 
                               "origin", "group", "coat", "colors", "description"]
            for field in required_fields:
                if field not in breed_info:
                    breed_info[field] = "Not specified" if field != "scientific_name" else "Canis lupus familiaris"
            
            # Add interesting fact if not present
            if "interesting_fact" not in breed_info:
                breed_info["interesting_fact"] = f"The {breed_name} is a unique and fascinating dog breed with its own special characteristics."
            
            # Create result dictionary with all the information
            result = {
                'label': breed_name,
                'confidence': confidence * 100,  # Keep this for backend code
                'confidence_internal': confidence * 100,  # Duplicate for internal use
                'scientific_name': breed_info['scientific_name'],
                'height': breed_info['height'],
                'weight': breed_info['weight'],
                'life_span': breed_info['life_span'],
                'temperament': breed_info['temperament'],
                'origin': breed_info['origin'],
                'group': breed_info['group'],
                'coat': breed_info['coat'],
                'colors': breed_info['colors'],
                'description': breed_info['description'],
                'interesting_fact': breed_info['interesting_fact'],
                'model_used_internal': "General" if use_general_model else "Specialized Dog Breed"  # Renamed
            }
            
            print(f"📊 Dog breed prediction result: {breed_name} ({confidence*100:.2f}%)")
            print(f"📊 Scientific name: {breed_info['scientific_name']}")
            print(f"📊 Origin: {breed_info['origin']}")
            print(f"📊 Model used: {'General' if use_general_model else 'Specialized Dog Breed'}")
            print(f"🐕🐕🐕 DOG BREED PREDICTION COMPLETE 🐕🐕🐕\n\n")
            
            return result
        
        except Exception as e:
            print(f"❌ Dog breed prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'label': 'Error during analysis',
                'confidence': 0.0,
                'description': f'An error occurred during dog breed analysis: {str(e)}. Please try again with a different image.'
            }

    def _preprocess_flower_image(self, image_path, input_shape=(224, 224)):
        """Special preprocessing for flower images to improve model accuracy."""
        try:
            from PIL import ImageEnhance, ImageOps
            
            # Open and resize image
            img = Image.open(image_path).convert('RGB')
            print(f"✅ Opened image: {image_path}, size: {img.size}")
            
            # Extra preprocessing for better results
            # 1. Center crop to square
            min_dim = min(img.size)
            img = ImageOps.fit(img, (min_dim, min_dim), method=Image.LANCZOS)
            
            # 2. Enhance color and contrast
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.3)  # Increase color saturation
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.3)  # Increase contrast
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.1)  # Slight brightness increase
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.5)  # Increase sharpness
            
            # 3. Resize to model input shape
            img = img.resize(input_shape, Image.LANCZOS)
            print(f"✅ Preprocessed image to size: {img.size}")
            
            # 4. Convert to array and normalize to 0-1 range 
            img_array = np.array(img) / 255.0
            
            # 5. Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            print(f"✅ Final array shape: {img_array.shape}")
            
            return img_array
            
        except Exception as e:
            print(f"❌ Error in flower image preprocessing: {e}")
            # Fall back to standard preprocessing
            return self._preprocess_keras(image_path, input_shape)

    def predict_flower(self, image_path: str) -> Optional[Dict[str, Union[str, float]]]:
        """
        Predict flower species from an image.
        
        Args:
            image_path: Path to the image file to classify
            
        Returns:
            Dictionary with prediction results or None if prediction failed
        """
        if self.flower_model is None or not self.flower_labels:
            print("⚠️ Flower model or labels not available")
            return None
            
        try:
            print(f"\n\n🌸🌸🌸 FLOWER PREDICTION STARTING 🌸🌸🌸")
            print(f"🌸 Starting flower prediction for image: {image_path}")
            
            # First validate the image exists
            if not os.path.exists(image_path):
                print(f"❌ Error: Image file not found: {image_path}")
                return {
                    'label': 'Image not found',
                    'confidence': 0.0,
                    'description': f'The image file could not be found at {image_path}'
                }
            
            # Use specialized preprocessing for flowers
            print("🔄 Using specialized flower image preprocessing")
            img_array = self._preprocess_flower_image(image_path, self.flower_input_shape)
            
            # Get predictions from the model
            print("🔄 Running flower species prediction")
            predictions = self.flower_model.predict(img_array)
            
            # Get the top 3 results for better analysis
            top_indices = np.argsort(-predictions[0])[:3]
            top_predictions = []
            
            print("🔍 Top 3 flower species predictions:")
            for i, idx in enumerate(top_indices):
                if idx < len(self.flower_labels):
                    label = self.flower_labels[idx]
                    confidence = float(predictions[0][idx])
                    top_predictions.append((label, confidence))
                    print(f"   {i+1}. {label} ({confidence:.4f} = {confidence*100:.2f}%)")
            
            # If no valid predictions, return error
            if not top_predictions:
                print("❌ No valid flower species predictions found")
                return {
                    'label': 'No flower species identified',
                    'confidence': 0.0,
                    'description': 'The model could not identify any flower species in this image. Try a clearer image of a flower.'
                }
            
            # Use the top prediction
            flower_name, confidence = top_predictions[0]
            matched_info = None
            
            # Try to find matching info for any of the top predictions
            for name, conf in top_predictions:
                # Try several variations to find a match in the flower_info dictionary
                variations = [
                    name.lower(),  # "tiger lily"
                    name.replace(' ', '_').lower(),  # "tiger_lily"
                    name.replace('-', ' ').lower(),  # convert "globe-flower" to "globe flower"
                    name.replace('\'', '').lower(),  # remove apostrophes
                    ' '.join(name.lower().split()),  # normalize whitespace
                ]
                
                for variation in variations:
                    if variation in self.flower_info:
                        print(f"✅ Found match in flower_info for '{name}' using variation: '{variation}'")
                        matched_info = self.flower_info[variation]
                        flower_name = name  # Use the matched name
                        confidence = conf   # Use the confidence from this match
                        break
                        
                if matched_info:
                    break  # Stop searching if we found a match
            
            # If still no match, try a more fuzzy match
            if not matched_info:
                print(f"⚠️ No exact match found for any top prediction in flower_info")
                print(f"⚠️ Available keys (first 5): {list(self.flower_info.keys())[:5] if self.flower_info else []}")
                # Try a fuzzy match by looking for partial matches
                for flower_key in self.flower_info.keys():
                    if flower_name.lower() in flower_key or flower_key in flower_name.lower():
                        print(f"✅ Found fuzzy match: '{flower_key}' for '{flower_name}'")
                        matched_info = self.flower_info[flower_key]
                        break
            
            # If still no match, use generic info
            if not matched_info:
                print(f"⚠️ No flower info match found after all attempts, using generic info")
                matched_info = {
                    "scientific_name": f"{flower_name.title()} sp.",
                    "origin": "Various regions",
                    "features": f"This flower has distinctive characteristics unique to the {flower_name} species. It typically displays vibrant colors and specific petal arrangements to attract pollinators."
                }
            
            # Create the result with matched info
            result = {
                'label': flower_name,
                'confidence': confidence,
                'scientific_name': matched_info.get('scientific_name', 'Unknown'),
                'origin': matched_info.get('origin', 'Unknown'),
                'features': matched_info.get('features', 'No features available')
            }
            
            print(f"📊 Flower prediction result: {flower_name} ({confidence*100:.2f}%)")
            print(f"📊 Features: '{result.get('features')[:50]}...'")
            print(f"🌸🌸🌸 FLOWER PREDICTION COMPLETE 🌸🌸🌸\n\n")
            return result
            
        except Exception as e:
            print(f"❌ Flower prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'label': 'Error during analysis',
                'confidence': 0.0,
                'description': f'An error occurred during flower species analysis: {str(e)}. Please try again with a different image.'
            }

    def _check_visual_similarities(self, image_path: str, predictions: List[Tuple[str, float]]) -> Optional[List[Tuple[str, float]]]:
        """
        Perform additional checks for visual similarities between commonly confused flowers.
        Returns corrected predictions if needed, otherwise returns None.
        """
        try:
            from PIL import Image, ImageStat
            
            # Load image
            img = Image.open(image_path).convert('RGB')
            
            # Get dominant colors
            stat = ImageStat.Stat(img)
            r, g, b = stat.mean
            
            # Sweet pea typically has more pink/purple while snapdragons vary more
            if "snapdragon" in [p[0].lower() for p in predictions]:
                # Calculate pink/purple dominance (high R, lower G, medium B)
                pink_purple_score = (r > 120 and g < 100 and b > 80)
                
                # Check for tendrils or thin stems characteristic of sweet peas
                # This is simplistic but could help in some cases
                # A more advanced approach would use edge detection
                
                # If it looks like a sweet pea, promote sweet pea in the predictions
                if pink_purple_score:
                    print("✅ Image has color characteristics of sweet pea (pink/purple)")
                    
                    # Check if sweet pea is already in predictions
                    sweet_pea_idx = None
                    for i, (name, _) in enumerate(predictions):
                        if "sweet pea" in name.lower() or "lathyrus" in name.lower():
                            sweet_pea_idx = i
                            break
                    
                    if sweet_pea_idx is not None:
                        # Promote sweet pea to top prediction
                        sweet_pea = predictions[sweet_pea_idx]
                        enhanced_conf = min(1.0, sweet_pea[1] * 1.3)  # Boost confidence by 30%
                        new_predictions = [(sweet_pea[0], enhanced_conf)] + [p for i, p in enumerate(predictions) if i != sweet_pea_idx]
                        return new_predictions
            
            return None
        except Exception as e:
            print(f"⚠️ Error in visual similarity check: {e}")
            return None
            
    def _check_for_sweet_pea(self, image_path: str) -> bool:
        """
        Check if an image has characteristics of a sweet pea based on color and structure.
        This is a simple heuristic that could be improved with more sophisticated image analysis.
        """
        try:
            from PIL import Image, ImageStat
            
            # Load image
            img = Image.open(image_path).convert('RGB')
            
            # Resize for consistency
            img = img.resize((224, 224))
            
            # Get color statistics
            stat = ImageStat.Stat(img)
            r, g, b = stat.mean
            
            # Sweet peas often have these characteristics:
            # 1. Pinks and purples are common (high R, medium to high B)
            pink_purple_dominant = (r > 100 and b > 80 and r > g)
            
            # 2. Sweet peas have more green foliage visible typically
            significant_green = (g > 80)
            
            # 3. Sweet peas often have a more balanced color distribution compared to snapdragons
            r_g_ratio = r / g if g > 0 else 999
            balanced_colors = (0.9 < r_g_ratio < 1.8)
            
            # Calculate an overall likelihood score
            sweet_pea_likelihood = (pink_purple_dominant and significant_green and balanced_colors)
            
            print(f"✅ Sweet pea likelihood check: {sweet_pea_likelihood}")
            print(f"   - Pink/purple dominant: {pink_purple_dominant}")
            print(f"   - Significant green: {significant_green}")
            print(f"   - Balanced colors: {balanced_colors}")
            
            return sweet_pea_likelihood
            
        except Exception as e:
            print(f"⚠️ Error checking for sweet pea: {e}")
            return False

    def predict_plant(self, image_path: str) -> Optional[Dict[str, Union[str, float]]]:
        """
        Predict plant/tree species from an image.
        
        Args:
            image_path: Path to the image file to classify
            
        Returns:
            Dictionary with prediction results or None if prediction failed
        """
        if self.plant_model is None:
            print("⚠️ Plant species model not available")
            return {
                'label': 'Plant model not available', 
                'confidence': 0.0,
                'description': 'The plant recognition model is not available. Please try again later.'
            }
            
        if not self.plant_labels:
            print("⚠️ Plant species labels not available")
            return {
                'label': 'Plant labels not available', 
                'confidence': 0.0,
                'description': 'The plant species labels are not available. Please try again later.'
            }
            
        try:
            print(f"\n\n🌿🌿🌿 PLANT SPECIES PREDICTION STARTING 🌿🌿🌿")
            print(f"🌿 Starting plant species prediction for image: {image_path}")
            
            # First validate the image exists
            if not os.path.exists(image_path):
                print(f"❌ Error: Image file not found: {image_path}")
                return {
                    'label': 'Image not found',
                    'confidence': 0.0,
                    'description': f'The image file could not be found at {image_path}'
                }
            
            # Preprocess the image for the plant model
            try:
                # Ensure proper preprocessing with the right input shape
                img_array = self._preprocess_keras(image_path, self.plant_input_shape)
                print(f"✅ Preprocessed image to shape {img_array.shape}")
            except Exception as e:
                print(f"❌ Error preprocessing image: {str(e)}")
                import traceback
                traceback.print_exc()
                return {
                    'label': 'Image Preprocessing Failed',
                    'confidence': 0.0,
                    'description': f'Failed to process the image for plant analysis: {str(e)}'
                }
            
            # Get predictions from the model
            print("🔄 Running plant species prediction with model")
            try:
                predictions = self.plant_model.predict(img_array, verbose=0)
                print(f"✅ Got predictions with shape {predictions.shape}")
            except Exception as e:
                print(f"❌ Error during model prediction: {str(e)}")
                import traceback
                traceback.print_exc()
                return {
                    'label': 'Prediction Failed',
                    'confidence': 0.0,
                    'description': f'The plant model failed to process the image: {str(e)}'
                }
            
            # Get the top 3 results for better analysis
            top_indices = np.argsort(-predictions[0])[:3]
            top_predictions = []
            
            print("🔍 Top 3 plant species predictions:")
            for i, idx in enumerate(top_indices):
                if idx < len(self.plant_labels):
                    label = self.plant_labels[idx]
                    confidence = float(predictions[0][idx])
                    top_predictions.append((label, confidence))
                    print(f"   {i+1}. {label} ({confidence:.4f} = {confidence*100:.2f}%)")
            
            # If no valid predictions, return error
            if not top_predictions:
                print("❌ No valid plant species predictions found")
                return {
                    'label': 'No plant species identified',
                    'confidence': 0.0,
                    'description': 'The model could not identify any plant species in this image. Try a clearer image of a plant or tree.'
                }
            
            # Use the top prediction
            species_id, confidence = top_predictions[0]
            
            # Find plant info from JSON
            species_info = None
            species_name = species_id  # Default to ID if no pretty name is found
            
            if self.plant_info:
                try:
                    # Try to find the species in the info dictionary
                    if species_id in self.plant_info:
                        species_info = self.plant_info[species_id]
                        species_name = species_info.get('common_name', species_id)
                        print(f"✅ Found exact match for '{species_id}' in plant_info")
                    else:
                        # Try with different formatting or partial matches
                        for info_species in self.plant_info.keys():
                            if species_id.lower() == info_species.lower() or \
                               species_id.lower().replace('_', '') == info_species.lower().replace('_', ''):
                                species_info = self.plant_info[info_species]
                                species_name = species_info.get('common_name', info_species)
                                print(f"✅ Found match after normalization: '{info_species}' for '{species_id}'")
                                break
                                
                        # If still no match, look for partial matches
                        if not species_info:
                            for info_species in self.plant_info.keys():
                                if species_id.lower() in info_species.lower() or info_species.lower() in species_id.lower():
                                    species_info = self.plant_info[info_species]
                                    species_name = species_info.get('common_name', info_species)
                                    print(f"✅ Found partial match: '{info_species}' for '{species_id}'")
                                    break
                except Exception as e:
                    print(f"⚠️ Error finding plant info: {e}")
            else:
                print("⚠️ Plant info dictionary is not available")
            
            # Make the species name more readable by replacing underscores with spaces and capitalizing
            if '_' in species_name:
                species_name = species_name.replace('_', ' ').title()
            
            # Create generic info if no match found
            if not species_info:
                print(f"⚠️ No species info found for '{species_id}', using generic info")
                scientific_name = species_id.replace('_', ' ').title()
                species_info = {
                    "scientific_name": scientific_name,
                    "common_name": species_name,
                    "family": "Unknown",
                    "origin": "Unknown",
                    "description": f"This appears to be {species_name} (scientific name: {scientific_name}). Unfortunately, detailed information about this specific plant species is not available in our database.",
                    "features": "Unknown"
                }
            
            # Create result dictionary with all the information
            result = {
                'label': species_name,
                'scientific_name': species_info.get('scientific_name', 'Unknown'),
                'confidence': confidence * 100,  # Convert to percentage
                'common_name': species_info.get('common_name', species_name),
                'family': species_info.get('family', 'Unknown'),
                'origin': species_info.get('origin', 'Unknown'),
                'description': species_info.get('description', f"A {species_name} plant."),
                'features': species_info.get('features', 'Unknown')
            }
            
            print(f"📊 Plant species prediction result: {species_name} ({confidence*100:.2f}%)")
            print(f"📊 Scientific name: {result.get('scientific_name')}")
            print(f"📊 Family: {result.get('family')}")
            print(f"📊 Origin: {result.get('origin')}")
            print(f"🌿🌿🌿 PLANT SPECIES PREDICTION COMPLETE 🌿🌿🌿\n\n")
            
            return result
            
        except Exception as e:
            print(f"❌ Plant species prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'label': 'Error during analysis',
                'confidence': 0.0,
                'description': f'An error occurred during plant species analysis: {str(e)}. Please try again with a different image.'
            }

    def predict_insect(self, image_path: str) -> Optional[Dict[str, Union[str, float]]]:
        """
        Predict insect species from an image.
        
        Args:
            image_path: Path to the image file to classify
            
        Returns:
            Dictionary with prediction results or None if prediction failed
        """
        if self.insect_model is None or not self.insect_labels:
            print("⚠️ Insect model or labels not available")
            return None
            
        try:
            print(f"🔍 Starting insect prediction for image: {image_path}")
            
            # Preprocess the image for the keras model
            preprocessed_image = self._preprocess_keras(image_path, self.insect_input_shape)
            
            # Get predictions from the model
            predictions = self.insect_model.predict(preprocessed_image)
            
            # Get the top class index (add 1 to match the 1-indexed labels in the insects.txt)
            class_idx = np.argmax(predictions[0]) + 1
            confidence = float(predictions[0][class_idx - 1])
            
            # Get the class name from the mapping
            class_name = self.insect_labels.get(str(class_idx), f"Unknown Insect ({class_idx})")
            
            print(f"🐛 Insect prediction: {class_name} (Confidence: {confidence:.4f})")
            print(f"🐛 Class index: {class_idx}, Found in labels: {str(class_idx) in self.insect_labels}")
            
            # Convert class name to match insect_info keys (replace spaces with underscores)
            info_key = class_name.lower().replace(' ', '_')
            
            # Get additional info about the insect if available
            additional_info = {}
            if info_key in self.insect_info:
                additional_info = self.insect_info[info_key]
                print(f"✅ Found additional information for {class_name} (key: {info_key})")
                print(f"✅ Available fields: {', '.join(additional_info.keys())}")
            else:
                print(f"⚠️ No additional information found for {class_name} (key: {info_key})")
                print(f"⚠️ Available keys in insect_info: {', '.join(list(self.insect_info.keys())[:5])}...")
                
            # Build the result
            result = {
                'class_name': class_name,
                'confidence': confidence,
                'additional_info': additional_info
            }
            
            print(f"🐛 Final result: {result['class_name']} ({confidence:.4f}), Info fields: {len(additional_info)}")
            
            # Ensure all necessary information is available in the result
            result = self._ensure_complete_info(result, 'insects')
            
            return result
            
        except Exception as e:
            print(f"⚠️ Error during insect prediction: {str(e)}")
            return None

    def predict_cat(self, image_path: str) -> Optional[Dict[str, Union[str, float]]]:
        """
        Predict cat breed from an image.
        
        Args:
            image_path: Path to the image file to classify
            
        Returns:
            Dictionary with prediction results or None if prediction failed
        """
        try:
            print(f"\n\n🐱🐱🐱 CAT PREDICTION STARTING 🐱🐱🐱")
            print(f"🐱 Starting cat breed prediction for image: {image_path}")
            
            # First validate the image exists
            if not os.path.exists(image_path):
                print(f"❌ Error: Image file not found: {image_path}")
                return {
                    'label': 'Image not found',
                    'confidence': 0.0,
                    'description': f'The image file could not be found at {image_path}'
                }
                
            # Check if we have a dedicated cat model
            if self.cat_model is not None and self.cat_labels:
                print("✅ Using dedicated cat breed classifier model")
                # Preprocess the image for the cat model
                print("🔄 Preprocessing image for cat breed classifier")
                img_array = self._preprocess_keras(image_path, self.cat_input_shape)
                
                # Get predictions from the model
                print("🔄 Running cat breed prediction")
                predictions = self.cat_model.predict(img_array)
                
                # Get the top 3 results for better analysis
                top_indices = np.argsort(-predictions[0])[:3]
                top_predictions = []
                
                print("🔍 Top 3 cat breed predictions:")
                for i, idx in enumerate(top_indices):
                    if idx < len(self.cat_labels):
                        label = self.cat_labels[idx]
                        confidence = float(predictions[0][idx])
                        top_predictions.append((label, confidence))
                        print(f"   {i+1}. {label} ({confidence:.4f} = {confidence*100:.2f}%)")
                
                # If no valid predictions, return error
                if not top_predictions:
                    print("❌ No valid cat breed predictions found")
                    return {
                        'label': 'No cat breed identified',
                        'confidence': 0.0,
                        'description': 'The model could not identify any cat breed in this image. Try a clearer image of a cat.'
                    }
                
                # Use the top prediction
                breed_id, confidence = top_predictions[0]
                
                # Load cat breed info from JSON if available
                cat_info_path = os.path.join(MODEL_DIR, 'cat_breed_info.json')
                breed_info = None
                breed_name = breed_id  # Default to ID if no pretty name is found
                
                # Make the breed name more readable by replacing underscores with spaces and capitalizing
                if '_' in breed_name:
                    breed_name = breed_name.replace('_', ' ').title()
                
                # Load cat breed info if available
                if os.path.exists(cat_info_path):
                    try:
                        with open(cat_info_path, 'r', encoding='utf-8') as f:
                            cat_info = json.load(f)
                        
                        # Try to find the breed in the info dictionary
                        if breed_id in cat_info:
                            breed_info = cat_info[breed_id]
                            breed_name = breed_info.get('name', breed_name)
                            print(f"✅ Found exact match for '{breed_id}' in cat_info")
                        else:
                            # Try with different formatting or partial matches
                            for info_breed in cat_info.keys():
                                if breed_id.lower() == info_breed.lower() or \
                                   breed_id.lower().replace('_', '') == info_breed.lower().replace('_', ''):
                                    breed_info = cat_info[info_breed]
                                    breed_name = breed_info.get('name', breed_name)
                                    print(f"✅ Found match after normalization: '{info_breed}' for '{breed_id}'")
                                    break
                                    
                            # If still no match, look for partial matches
                            if not breed_info:
                                for info_breed in cat_info.keys():
                                    if breed_id.lower() in info_breed.lower() or info_breed.lower() in breed_id.lower():
                                        breed_info = cat_info[info_breed]
                                        breed_name = breed_info.get('name', breed_name)
                                        print(f"✅ Found partial match: '{info_breed}' for '{breed_id}'")
                                        break
                    except Exception as e:
                        print(f"⚠️ Error loading cat breed info: {e}")
                    
                    # Create generic info if no match found
                    if not breed_info:
                        print(f"⚠️ No breed info found for '{breed_id}', using generic info")
                        breed_info = {
                            "name": breed_name,
                            "origin": "Unknown",
                            "temperament": "Unknown",
                            "weight": "Unknown",
                            "life_span": "Unknown",
                            "description": f"This appears to be a {breed_name} cat. Unfortunately, detailed information about this specific cat breed is not available in our database."
                        }
                    
                    # Create result dictionary with all the information
                    result = {
                        'label': breed_name,
                        'confidence': confidence,
                        'origin': breed_info.get('origin', 'Unknown'),
                        'temperament': breed_info.get('temperament', 'Unknown'),
                        'weight': breed_info.get('weight', 'Unknown'),
                        'life_span': breed_info.get('life_span', 'Unknown'),
                        'description': breed_info.get('description', f"A {breed_name} cat.")
                    }
                else:
                    # Fallback to general model for cat detection
                    print("⚠️ Dedicated cat model not available, using general classification model")
                    
                    # We'll use the general classification model as we don't have a specific cat model
                    if not self.model and not self.general_model:
                        print("❌ Error: General classification model not loaded")
                        return {
                            'label': 'Model not available',
                            'confidence': 0.0,
                            'description': 'The image classification model could not be loaded.'
                        }
                    
                    # Use the general prediction as a base
                    general_result = self.predict_general(image_path)
                    
                    if not general_result:
                        print("❌ General prediction failed")
                        return {
                            'label': 'Classification failed',
                            'confidence': 0.0,
                            'description': 'The system could not classify this image.'
                        }
                    
                    # Extract prediction details
                    label = general_result.get('label', '').lower()
                    confidence = general_result.get('confidence', 0.0)
                    
                    print(f"🔍 General prediction result: {label} ({confidence:.2f})")
                    
                    # Check if the prediction is related to cats
                    cat_keywords = ['cat', 'kitten', 'feline', 'tabby', 'persian', 'siamese', 'bengal', 'maine coon', 
                                   'ragdoll', 'abyssinian', 'british shorthair', 'egyptian']
                    
                    is_cat = False
                    for keyword in cat_keywords:
                        if keyword in label:
                            is_cat = True
                            print(f"✅ Detected cat-related keyword: '{keyword}' in '{label}'")
                            break
                    
                    if not is_cat:
                        print(f"⚠️ No cat detected in image (label: {label})")
                        return {
                            'label': 'Not a cat',
                            'confidence': 0.0,
                            'description': 'The image does not appear to contain a cat.'
                        }
                    
                    # Try to match with a cat breed from our cat_info dictionary
                    matched_info = None
                    matched_key = None
                    
                    # Clean up the label
                    clean_label = label.replace('_', ' ').split(',')[0].lower()
                    
                    # Try different variations to match with cat_info
                    variations = [
                        clean_label,
                        clean_label + " cat",
                        clean_label.replace('cat', '').strip(),
                    ]
                    
                    # Add specific keyword mappings for known ImageNet cat classes
                    if 'tiger cat' in clean_label or 'tabby' in clean_label:
                        variations.append('tabby cat')
                    if 'egyptian' in clean_label:
                        variations.append('egyptian cat')
                    
                    print(f"🔍 Trying to match with variations: {variations}")
                    
                    # Try direct matches first
                    for variation in variations:
                        if variation in self.cat_info:
                            matched_info = self.cat_info[variation]
                            matched_key = variation
                            print(f"✅ Found exact match in cat_info: {variation}")
                            break
                    
                    # If no direct match, try fuzzy matching
                    if not matched_info:
                        print(f"⚠️ No exact match found for '{clean_label}' in cat_info")
                        best_match_score = 0
                        
                        for cat_key in self.cat_info.keys():
                            # Try various matching strategies
                            for variation in variations:
                                # Check if variation is in cat_key or vice versa
                                if variation in cat_key or cat_key in variation:
                                    # Calculate match score based on length of common substring
                                    match_score = len(variation) / max(len(variation), len(cat_key)) if variation else 0
                                    
                                    if match_score > best_match_score:
                                        best_match_score = match_score
                                        matched_info = self.cat_info[cat_key]
                                        matched_key = cat_key
                        
                        if matched_key:
                            print(f"✅ Found fuzzy match: '{matched_key}' for '{clean_label}' (score: {best_match_score:.2f})")
                    
                    # If still no match, use generic cat info
                    if not matched_info:
                        print(f"⚠️ No cat breed match found after all attempts, using generic cat info")
                        if 'cat' in self.cat_info:
                            matched_info = self.cat_info['cat']
                            matched_key = 'cat'
                            print(f"✅ Using generic cat information")
                        else:
                            # Create basic info if even generic cat info is missing
                            matched_info = {
                                "scientific_name": "Felis catus",
                                "description": "The domestic cat is a small, typically furry, carnivorous mammal. They are often called house cats when kept as indoor pets or simply cats when there is no need to distinguish them from other felids and felines.",
                                "origin": "Worldwide domestication",
                                "lifespan": "12-18 years",
                                "weight": "3.6-4.5 kg",
                                "height": "23-25 cm",
                                "temperament": "Varies widely"
                            }
                            matched_key = 'domestic cat'
                    
                    # Format the breed name for display
                    breed_name = matched_key.replace('_', ' ').title()
                    
                    # Create the final result
                    result = {
                        'label': breed_name,
                        'confidence': confidence,
                        'scientific_name': matched_info.get('scientific_name', 'Felis catus'),
                        'description': matched_info.get('description', 'No description available'),
                        'origin': matched_info.get('origin', 'Unknown'),
                        'lifespan': matched_info.get('lifespan', 'Unknown'),
                        'weight': matched_info.get('weight', 'Unknown'),
                        'height': matched_info.get('height', 'Unknown'),
                        'temperament': matched_info.get('temperament', 'Unknown')
                    }
            else:
                # Fallback to general model for cat detection
                print("⚠️ Dedicated cat model not available, using general classification model")
                
                # We'll use the general classification model as we don't have a specific cat model
                if not self.model and not self.general_model:
                    print("❌ Error: General classification model not loaded")
                    return {
                        'label': 'Model not available',
                        'confidence': 0.0,
                        'description': 'The image classification model could not be loaded.'
                    }
                
                # Use the general prediction as a base
                general_result = self.predict_general(image_path)
                
                if not general_result:
                    print("❌ General prediction failed")
                    return {
                        'label': 'Classification failed',
                        'confidence': 0.0,
                        'description': 'The system could not classify this image.'
                    }
                
                # Extract prediction details
                label = general_result.get('label', '').lower()
                confidence = general_result.get('confidence', 0.0)
                
                print(f"🔍 General prediction result: {label} ({confidence:.2f})")
                
                # Check if the prediction is related to cats
                cat_keywords = ['cat', 'kitten', 'feline', 'tabby', 'persian', 'siamese', 'bengal', 'maine coon', 
                               'ragdoll', 'abyssinian', 'british shorthair', 'egyptian']
                
                is_cat = False
                for keyword in cat_keywords:
                    if keyword in label:
                        is_cat = True
                        print(f"✅ Detected cat-related keyword: '{keyword}' in '{label}'")
                        break
                
                if not is_cat:
                    print(f"⚠️ No cat detected in image (label: {label})")
                    return {
                        'label': 'Not a cat',
                        'confidence': 0.0,
                        'description': 'The image does not appear to contain a cat.'
                    }
                
                # Try to match with a cat breed from our cat_info dictionary
                matched_info = None
                matched_key = None
                
                # Clean up the label
                clean_label = label.replace('_', ' ').split(',')[0].lower()
                
                # Try different variations to match with cat_info
                variations = [
                    clean_label,
                    clean_label + " cat",
                    clean_label.replace('cat', '').strip(),
                ]
                
                # Add specific keyword mappings for known ImageNet cat classes
                if 'tiger cat' in clean_label or 'tabby' in clean_label:
                    variations.append('tabby cat')
                if 'egyptian' in clean_label:
                    variations.append('egyptian cat')
                
                print(f"🔍 Trying to match with variations: {variations}")
                
                # Try direct matches first
                for variation in variations:
                    if variation in self.cat_info:
                        matched_info = self.cat_info[variation]
                        matched_key = variation
                        print(f"✅ Found exact match in cat_info: {variation}")
                        break
                
                # If no direct match, try fuzzy matching
                if not matched_info:
                    print(f"⚠️ No exact match found for '{clean_label}' in cat_info")
                    best_match_score = 0
                    
                    for cat_key in self.cat_info.keys():
                        # Try various matching strategies
                        for variation in variations:
                            # Check if variation is in cat_key or vice versa
                            if variation in cat_key or cat_key in variation:
                                # Calculate match score based on length of common substring
                                match_score = len(variation) / max(len(variation), len(cat_key)) if variation else 0
                                
                                if match_score > best_match_score:
                                    best_match_score = match_score
                                    matched_info = self.cat_info[cat_key]
                                    matched_key = cat_key
                    
                    if matched_key:
                        print(f"✅ Found fuzzy match: '{matched_key}' for '{clean_label}' (score: {best_match_score:.2f})")
                
                # If still no match, use generic cat info
                if not matched_info:
                    print(f"⚠️ No cat breed match found after all attempts, using generic cat info")
                    if 'cat' in self.cat_info:
                        matched_info = self.cat_info['cat']
                        matched_key = 'cat'
                        print(f"✅ Using generic cat information")
                    else:
                        # Create basic info if even generic cat info is missing
                        matched_info = {
                            "scientific_name": "Felis catus",
                            "description": "The domestic cat is a small, typically furry, carnivorous mammal. They are often called house cats when kept as indoor pets or simply cats when there is no need to distinguish them from other felids and felines.",
                            "origin": "Worldwide domestication",
                            "lifespan": "12-18 years",
                            "weight": "3.6-4.5 kg",
                            "height": "23-25 cm",
                            "temperament": "Varies widely"
                        }
                        matched_key = 'domestic cat'
                
                # Format the breed name for display
                breed_name = matched_key.replace('_', ' ').title()
                
                # Create the final result
                result = {
                    'label': breed_name,
                    'confidence': confidence,
                    'scientific_name': matched_info.get('scientific_name', 'Felis catus'),
                    'description': matched_info.get('description', 'No description available'),
                    'origin': matched_info.get('origin', 'Unknown'),
                    'lifespan': matched_info.get('lifespan', 'Unknown'),
                    'weight': matched_info.get('weight', 'Unknown'),
                    'height': matched_info.get('height', 'Unknown'),
                    'temperament': matched_info.get('temperament', 'Unknown')
                }
            
            print(f"📊 Cat prediction result: {result['label']} ({confidence*100:.2f}%)")
            print(f"🐱🐱🐱 CAT PREDICTION COMPLETE 🐱🐱🐱\n\n")
            return result
            
        except Exception as e:
            print(f"❌ Cat prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'label': 'Error during analysis',
                'confidence': 0.0,
                'description': f'An error occurred during cat breed analysis: {str(e)}. Please try again with a different image.'
            }

    # --- Helper Functions ---
    def _preprocess_pytorch(self, image_path):
        """Preprocesses image for PyTorch ResNet."""
        if not self.general_model:
            raise ValueError("General PyTorch model not loaded")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = Image.open(image_path).convert('RGB')
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0)
        return batch_t
 
    def _preprocess_keras(self, image_path, input_shape=(224, 224)):
        """Preprocesses an image for Keras models with proper error handling."""
        try:
            print(f"🔄 Preprocessing image: {image_path} to shape {input_shape}")
            
            # Check if the file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Check file size
            file_size = os.path.getsize(image_path) / 1024  # size in KB
            print(f"📊 Image file size: {file_size:.2f} KB")
            
            if file_size < 1:
                print("⚠️ Warning: Image file is very small (<1KB)")
            
            # Open and preprocess the image
            img = Image.open(image_path).convert('RGB')
            print(f"📊 Original image size: {img.size}")
            
            # Resize image to input shape
            img = img.resize(input_shape, Image.LANCZOS)
            print(f"📊 Resized image to: {img.size}")
            
            # Convert image to numpy array
            img_array = np.array(img)
            print(f"📊 Converted to array with shape: {img_array.shape}")
            
            # Normalize pixel values to 0-1
            img_array = img_array / 255.0
            
            # Add batch dimension for model input
            img_array = np.expand_dims(img_array, axis=0)
            print(f"📊 Final preprocessed array shape: {img_array.shape}")
            
            return img_array
        except Exception as e:
            print(f"❌ Error in _preprocess_keras: {str(e)}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Failed to preprocess image {image_path} for Keras: {e}")
 
    # --- Main Prediction Method ---
    def predict(self, image_path: str, category: str = 'general') -> List[Dict[str, Union[str, float]]]:
        """
        Main prediction method that routes to the appropriate specialized model
        based on category and returns a standardized results format.
        
        Args:
            image_path: Path to the image file
            category: The category of the prediction (birds, plants, dogs, flowers, insects, etc.)
                      Use 'all' to run all available classifiers
            
        Returns:
            List of dictionaries containing prediction results
        """
        print(f"\n🔍 Predicting {category} for image: {image_path}")
        results = []
        
        # First check if the image exists
        if not os.path.exists(image_path):
            print(f"⚠️ Image not found at {image_path}")
            return [{"label": "Image not found", "confidence": 0.0, "error": f"Image not found at {image_path}"}]
        
        # Verify image integrity
        try:
            img = Image.open(image_path)
            img.verify()  # Verify image integrity
            print(f"✅ Image verified: {image_path}")
        except Exception as img_err:
            print(f"⚠️ Invalid or corrupted image: {str(img_err)}")
            return [{"label": "Unable to process image", "confidence": 0.0, "error": str(img_err)}]
            
        # Handle special 'all' category to run all available classifiers
        if category.lower() == 'all':
            print("🔄 Running all available classifiers")
            all_results = []
            
            # Try bird prediction if model is available
            if self.bird_model is not None:
                print("🔄 Running bird species classifier")
                bird_result = self.predict_bird(image_path)
                if bird_result:
                    bird_result = self._ensure_complete_info(bird_result, 'birds')
                    bird_result['category'] = 'birds'
                    all_results.append(bird_result)
            
            # Try dog prediction if model is available
            if self.dog_model is not None:
                print("🔄 Running dog breed classifier")
                dog_result = self.predict_dog(image_path)
                if dog_result:
                    dog_result = self._ensure_complete_info(dog_result, 'dogs')
                    dog_result['category'] = 'dogs'
                    all_results.append(dog_result)
            
            # Try cat prediction (uses general model)
            print("🔄 Running cat breed classifier")
            cat_result = self.predict_cat(image_path)
            if cat_result:
                cat_result = self._ensure_complete_info(cat_result, 'cats')
                cat_result['category'] = 'cats'
                all_results.append(cat_result)
            
            # Try flower prediction if model is available
            if self.flower_model is not None:
                print("🔄 Running flower classifier")
                flower_result = self.predict_flower(image_path)
                if flower_result:
                    flower_result = self._ensure_complete_info(flower_result, 'flowers')
                    flower_result['category'] = 'flowers'
                    all_results.append(flower_result)
            
            # Try plant prediction if model is available
            if self.plant_model is not None:
                print("🔄 Running plant classifier")
                plant_result = self.predict_plant(image_path)
                if plant_result:
                    plant_result = self._ensure_complete_info(plant_result, 'plants')
                    plant_result['category'] = 'plants'
                    all_results.append(plant_result)
            
            # Try insect prediction if model is available
            if self.insect_model is not None:
                print("🔄 Running insect classifier")
                insect_result = self.predict_insect(image_path)
                if insect_result:
                    # Format to be consistent with other results
                    formatted_result = {
                        'label': insect_result['class_name'],
                        'confidence': insect_result['confidence'],
                        'category': 'insects',
                        'additional_info': insect_result['additional_info']
                    }
                    formatted_result = self._ensure_complete_info(formatted_result, 'insects')
                    all_results.append(formatted_result)
            
            # Always add general prediction
            print("🔄 Running general classifier")
            general_result = self.predict_general(image_path)
            if general_result:
                general_result['category'] = 'general'
                all_results.append(general_result)
            
            # Return all results or error if none
            if all_results:
                print(f"✅ Completed all predictions: found {len(all_results)} results")
                return all_results
            else:
                print("⚠️ No results from any classifier")
                return [{"label": "No classification results", "confidence": 0.0}]
        
        # Handle specific category predictions
        try:
            if category.lower() == 'birds':
                print("🔄 Using bird species classifier")
                if self.bird_model is None:
                    print("⚠️ Bird model not loaded")
                    return [{"label": "Bird model not available", "confidence": 0.0}]
                    
                bird_result = self.predict_bird(image_path)
                if bird_result:
                    bird_result = self._ensure_complete_info(bird_result, 'birds')
                    results = [bird_result]
                
            elif category.lower() == 'dogs':
                print("🔄 Using dog breed classifier")
                if self.dog_model is None:
                    print("⚠️ Dog model not loaded")
                    return [{"label": "Dog model not available", "confidence": 0.0}]
                    
                dog_result = self.predict_dog(image_path)
                if dog_result:
                    dog_result = self._ensure_complete_info(dog_result, 'dogs')
                    results = [dog_result]
                    
            elif category.lower() == 'cats':
                print("🔄 Using cat breed classifier")
                cat_result = self.predict_cat(image_path)
                if cat_result:
                    cat_result = self._ensure_complete_info(cat_result, 'cats')
                    results = [cat_result]
                    
            elif category.lower() == 'flowers':
                print("🔄 Using flower classifier")
                if self.flower_model is None:
                    print("⚠️ Flower model not loaded")
                    return [{"label": "Flower model not available", "confidence": 0.0}]
                    
                flower_result = self.predict_flower(image_path)
                if flower_result:
                    flower_result = self._ensure_complete_info(flower_result, 'flowers')
                    results = [flower_result]
            
            elif category.lower() == 'plants':
                print("🔄 Using plant classifier")
                if self.plant_model is None:
                    print("⚠️ Plant model not loaded")
                    return [{"label": "Plant model not available", "confidence": 0.0}]
                    
                plant_result = self.predict_plant(image_path)
                if plant_result:
                    plant_result = self._ensure_complete_info(plant_result, 'plants')
                    results = [plant_result]
                    
            elif category.lower() == 'insects':
                print("🔄 Using insect classifier")
                if self.insect_model is None:
                    print("⚠️ Insect model not loaded")
                    return [{"label": "Insect model not available", "confidence": 0.0}]
                    
                insect_result = self.predict_insect(image_path)
                if insect_result:
                    # Format to be consistent with other results
                    formatted_result = {
                        'label': insect_result['class_name'],
                        'confidence': insect_result['confidence'],
                        'additional_info': insect_result.get('additional_info', {})
                    }
                    formatted_result = self._ensure_complete_info(formatted_result, 'insects')
                    results = [formatted_result]
                    
            else:
                # Default to general image classification
                print("🔄 Using general classifier")
                if self.general_model is None and self.model is None:
                    print("⚠️ General model not loaded")
                    return [{"label": "General model not available", "confidence": 0.0}]
                    
                general_result = self.predict_general(image_path)
                if general_result:
                    results = [general_result]
                    
            # Return the results if any were found
            if results:
                print(f"✅ Prediction successful: {results[0].get('label', 'Unknown')} ({results[0].get('confidence', 0.0):.2f})")
                return results
            else:
                print("⚠️ No predictions returned")
                return [{"label": "Object not recognized", "confidence": 0.0, 
                        "description": "The system could not identify this image with confidence. Try a clearer image or a different category."}]
                
        except Exception as e:
            print(f"⚠️ Error in prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return [{"label": "Error during prediction", "confidence": 0.0, "error": str(e)}]

    # --- Helper function to ensure complete information ---
    def _ensure_complete_info(self, result, category):
        """
        Ensures prediction results have complete information by adding fallback data for any missing fields.
        
        Args:
            result: The prediction result dictionary
            category: The category of the prediction (birds, plants, insects, etc.)
            
        Returns:
            Updated result dictionary with fallback information for missing fields
        """
        if not result:
            return result
            
        # Generic fallback values
        fallbacks = {
            "scientific_name": f"{result.get('label', 'Unknown species')} (scientific name)",
            "description": f"This {category[:-1] if category.endswith('s') else category} belongs to the {result.get('label', 'unknown')} species.",
            "interesting_fact": f"This {category[:-1] if category.endswith('s') else category} has unique characteristics that make it fascinating to study."
        }
        
        # Category-specific fallbacks
        if category == 'birds':
            bird_name = result.get('label', 'bird')
            category_fallbacks = {
                "height": "15-25 cm (6-10 inches)",
                "weight": "20-150 grams depending on the species",
                "wingspan": "25-40 cm (10-16 inches)",
                "lifespan": "4-8 years in the wild, up to 15 years in captivity",
                "habitat": "Forests, wetlands, grasslands, and urban environments",
                "diet": "Seeds, insects, fruits, nectar, and small vertebrates depending on the species",
                "description": f"The {bird_name} is characterized by its distinctive plumage, beak shape, and behavior adapted to its ecological niche.",
                "interesting_fact": f"The {bird_name} is known for its unique vocalizations and nesting behaviors that have evolved over thousands of years."
            }
            fallbacks.update(category_fallbacks)
            
        elif category == 'insects':
            # For insects, the data is nested in additional_info
            if 'additional_info' not in result:
                result['additional_info'] = {}
                
            insect_fallbacks = {
                "Scientific Name": result.get('scientific_name', f"{result.get('label', 'Unknown')} sp."),
                "Habitat": "Various natural habitats including forests, grasslands, and gardens",
                "Diet": "Varies by species, may include plant matter, nectar, or other insects",
                "Lifespan": "Typically a few weeks to months, depending on species",
                "Description": f"The {result.get('label', 'insect')} is characterized by its distinctive appearance and behaviors adapted to its ecological niche.",
                "Interesting Fact": f"The {result.get('label', 'insect')} plays an important role in its ecosystem and has evolved specialized adaptations."
            }
            
            # Add any missing fields to additional_info
            for key, value in insect_fallbacks.items():
                if key not in result['additional_info'] or not result['additional_info'][key]:
                    result['additional_info'][key] = value
                    
        elif category == 'plants':
            plant_fallbacks = {
                "common_name": result.get('label', 'Unknown plant'),
                "family": "Botanical classification family",
                "origin": "Native to various regions globally",
                "features": f"The {result.get('label', 'plant')} has distinctive features that help identify it in the wild."
            }
            fallbacks.update(plant_fallbacks)
            
        elif category == 'flowers':
            flower_fallbacks = {
                "origin": "Found in various regions globally",
                "features": f"The {result.get('label', 'flower')} is known for its distinctive appearance and growth characteristics."
            }
            fallbacks.update(flower_fallbacks)
            
        elif category == 'dogs':
            dog_name = result.get('label', 'dog')
            dog_fallbacks = {
                "height": "35-65 cm (14-26 inches) at the shoulder",
                "weight": "7-40 kg (15-90 pounds)",
                "life_span": "10-14 years on average",
                "temperament": "Loyal, social, and adaptable to family environments",
                "origin": "Domesticated from wolves thousands of years ago",
                "group": "Domestic dog breed",
                "coat": "Short to medium-length, varying in texture and density",
                "colors": "Various patterns and colors including black, brown, white, and tan"
            }
            fallbacks.update(dog_fallbacks)
            
        elif category == 'cats':
            cat_name = result.get('label', 'cat')
            cat_fallbacks = {
                "origin": "Domesticated from wildcat ancestors about 10,000 years ago",
                "lifespan": "12-16 years on average, up to 20+ years with proper care",
                "temperament": "Independent, curious, and territorial with varying degrees of sociability",
                "weight": "3.5-5.5 kg (8-12 pounds) for adult cats",
                "height": "23-25 cm (9-10 inches) at the shoulder",
                "coat": "Short to long, requires regular grooming depending on length",
                "colors": "Wide variety including tabby, solid, bi-color, and calico patterns"
            }
            fallbacks.update(cat_fallbacks)
            
        # Add fallbacks for any missing fields (except for insects which were handled separately)
        if category != 'insects':
            for key, value in fallbacks.items():
                if key not in result or not result[key]:
                    result[key] = value
                    
        return result
