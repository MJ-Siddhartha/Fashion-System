import os
import uuid
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import requests
from io import BytesIO
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import ast
import json
import re
from enum import Enum
from transformers import DistilBertTokenizer, DistilBertModel
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, silhouette_score
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import networkx as nx
from sklearn.cluster import DBSCAN
from PIL import Image
import validators
import gradio as gr
import base64
from io import BytesIO
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fashion_system.log'),
        logging.StreamHandler()
    ]
)

def parse_json_safely(json_string):
    """Safely parse JSON with error handling"""
    try:
        # Remove any BOM and clean the JSON string
        cleaned_json = json_string.strip().replace('\ufeff', '')
        return json.loads(cleaned_json)  # Use json.loads directly instead of recursive call
    except json.JSONDecodeError as e:
        logging.warning(f"Failed to parse JSON: {json_string[:50]}... Error: {str(e)}")
        return None

class ProductCategory(Enum):
    BATHROOM_VANITY = "bathroom_vanity"
    DRESS = "dress"
    EARRING = "earring"
    JEAN = "jean"
    SAREE = "saree"
    SHIRT = "shirt"
    SNEAKER = "sneaker"
    TSHIRT = "tshirt"
    WATCH = "watch"

class ProductData:
    def __init__(self, product_id: str, name: str, brand: str, price: float, 
                 category: ProductCategory, style_attributes: Dict, 
                 features: List, feature_image_s3: str = None):
        self.product_id = product_id
        self.name = name
        self.brand = brand
        self.price = price
        self.category = category
        self.style_attributes = style_attributes
        self.features = features
        self.feature_image_s3 = feature_image_s3

class DataProcessor:
    def __init__(self, image_dir='fashion_images'):
        self.price_pattern = re.compile(r'[^\d.]')
        self.image_dir = os.path.join(os.getcwd(), image_dir)
        
        # Create local image directory if it doesn't exist
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
            logging.info(f"Created local image directory: {self.image_dir}")

    def get_image_path(self, product_id):
        """Generate local file path instead of S3 URL"""
        return os.path.join(self.image_dir, f"{product_id}.jpg")

    def clean_price(self, price: Any) -> float:
        if isinstance(price, (int, float)):
            return float(price)
        try:
            return float(self.price_pattern.sub('', str(price))) or 0.0
        except:
            return 0.0

    def parse_style_attrs(self, attrs: Any) -> Dict:
        if isinstance(attrs, dict):
            return attrs
        try:
            return eval(str(attrs)) if isinstance(attrs, str) else {}
        except:
            return {}

    def process_row(self, row: Dict, category: ProductCategory) -> ProductData:
        try:
            style_attrs = self.parse_style_attrs(row.get('style_attributes'))
            product_id = row.get('product_id', '')
            
            # Use local path instead of S3 URL
            feature_image_path = self.get_image_path(product_id)
            
            return ProductData(
                product_id=product_id,
                name=row.get('product_name', ''),
                brand=row.get('brand', ''),
                price=self.clean_price(row.get('mrp')),
                category=category,
                style_attributes=style_attrs,
                features=self.parse_features(row.get('feature_list', [])),
                feature_image_s3=feature_image_path
            )
        except Exception as e:
            logging.error(f"Error processing row: {str(e)}")
            return None

    def parse_features(self, features: Any) -> List[str]:
        if isinstance(features, list):
            return features
        try:
            return eval(str(features)) if isinstance(features, str) else []
        except:
            return []

class FashionDataProcessor:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.processor = DataProcessor()

    def process_file(self, filename: str, category: ProductCategory) -> pd.DataFrame:
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath)
        processed = []

        for _, row in df.iterrows():
            try:
                product = self.processor.process_row(row.to_dict(), category)
                processed.append(vars(product))
            except Exception as e:
                logging.error(f"Error processing row: {str(e)}")
                continue

        return pd.DataFrame(processed)

    def process_all(self) -> pd.DataFrame:
        file_map = {
            ProductCategory.BATHROOM_VANITY: "Bathroom Vanities Data Dump.csv",
            ProductCategory.DRESS: "Dresses Data Dump.csv",
            ProductCategory.EARRING: "Earrings Data Dump.csv",
            ProductCategory.JEAN: "Jeans Data Dump.csv",
            ProductCategory.SAREE: "Saree Data Dump.csv",
            ProductCategory.SHIRT: "shirts_data_dump.csv",
            ProductCategory.SNEAKER: "Sneakers Data Dump.csv",
            ProductCategory.TSHIRT: "Tshirts Data Dump.csv",
            ProductCategory.WATCH: "Watches Data Dump.csv"
        }

        combined_data = []
        for category, filename in file_map.items():
            try:
                category_data = self.process_file(filename, category)
                combined_data.append(category_data)
            except Exception as e:
                logging.error(f"Error processing {category}: {str(e)}")

        return pd.concat(combined_data, ignore_index=True)

class FashionImageProcessor:
    """Handles image processing and feature extraction using GPU acceleration."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = models.efficientnet_b0(pretrained=True).to(device)
        self.model.eval()
        
        # Remove classification layer to get features
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def extract_image_features(self, image_url):
        """Extract features from image URL using EfficientNet."""
        try:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content)).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(img_tensor)
                
            return features.squeeze().cpu().numpy()
        except Exception as e:
            logging.error(f"Error processing image {image_url}: {str(e)}")
            return None

class TextFeatureExtractor:
    """Handles text processing and feature extraction using DistilBERT."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
        self.model.eval()
    
    def extract_text_features(self, text):
        """Extract features from text using DistilBERT."""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", 
                                  truncation=True, max_length=512,
                                  padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Use [CLS] token embedding as text representation
            return outputs.last_hidden_state[:, 0, :].cpu().numpy()
        except Exception as e:
            logging.error(f"Error processing text: {str(e)}")
            return None

class FashionOntology:
    """Manages the fashion ontology and its continuous evolution."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.feature_clusters = {}
        
    def build_initial_ontology(self, combined_features):
        """Build initial ontology structure from extracted features."""
        # Use DBSCAN to cluster similar features
        clustering = DBSCAN(eps=0.5, min_samples=5)
        clusters = clustering.fit_predict(combined_features)
        
        # Create nodes for each cluster
        for cluster_id in np.unique(clusters):
            if cluster_id != -1:  # Ignore noise points
                cluster_features = combined_features[clusters == cluster_id]
                centroid = np.mean(cluster_features, axis=0)
                self.feature_clusters[cluster_id] = centroid
                self.graph.add_node(f"cluster_{cluster_id}", 
                                  features=centroid,
                                  creation_date=datetime.now())
    
    def update_ontology(self, new_features):
        """Update ontology with new features, detecting emerging trends."""
        # Find closest existing cluster for each new feature
        for feature in new_features:
            distances = {
                cluster_id: np.linalg.norm(feature - centroid)
                for cluster_id, centroid in self.feature_clusters.items()
            }
            
            closest_cluster = min(distances.items(), key=lambda x: x[1])
            
            # If distance is too large, create new cluster
            if closest_cluster[1] > 0.7:
                new_cluster_id = len(self.feature_clusters)
                self.feature_clusters[new_cluster_id] = feature
                self.graph.add_node(f"cluster_{new_cluster_id}",
                                  features=feature,
                                  creation_date=datetime.now())
                logging.info(f"New trend detected: cluster_{new_cluster_id}")

    def evaluate_clustering(self, features):
        """Evaluate the quality of the current clustering."""
        cluster_labels = []
        for feature in features:
            # Find closest cluster
            distances = {
                cluster_id: np.linalg.norm(feature - centroid)
                for cluster_id, centroid in self.feature_clusters.items()
            }
            closest_cluster = min(distances.items(), key=lambda x: x[1])[0]
            cluster_labels.append(closest_cluster)
        
        return np.array(cluster_labels)

class FashionIntelligenceSystem:
    """Main system orchestrating all components and managing the continuous learning loop."""
    
    def __init__(self, data_directory):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.preprocessor = FashionDataProcessor(data_directory)
        self.image_processor = FashionImageProcessor(self.device)
        self.text_processor = TextFeatureExtractor(self.device)
        self.evaluator = EvaluationMetrics()
        self.evaluation_results = {}
        self.ontology = FashionOntology()
        
    def train(self):
        """Train the system on the initial dataset."""
        logging.info("Starting training process...")
        
        # Load and preprocess data
        data = self.preprocessor.process_all()
        print(data.columns)

        if 'feature_image_s3' not in data.columns:
            logging.warning("'feature_image_s3' column is missing in the data. Skipping related operations.")
            # If this column is essential, raise an error or handle accordingly.
            return
        
        # Process images and text in batches
        batch_size = 32
        all_features = []
        
        for i in tqdm(range(0, len(data), batch_size), desc="Processing items"):
            batch = data.iloc[i:i+batch_size]
            batch_features = []
            
            # Extract image features
            for url in batch['feature_image_s3']:
                img_features = self.image_processor.extract_image_features(url)
                if img_features is not None:
                    batch_features.append(img_features)
            
            # Extract text features
            for desc in batch['description']:
                text_features = self.text_processor.extract_text_features(desc)
                if text_features is not None:
                    batch_features.extend(text_features)
            
            if batch_features:
                all_features.extend(batch_features)
        
        # Build ontology
        self.ontology.build_initial_ontology(np.array(all_features))
        logging.info("Training completed successfully")
    
    def process_new_item(self, image_url, description):
        """Process a new fashion item and classify it."""
        # Extract features
        img_features = self.image_processor.extract_image_features(image_url)
        text_features = self.text_processor.extract_text_features(description)
        
        if img_features is None or text_features is None:
            return None
        
        # Combine features
        combined_features = np.concatenate([img_features, text_features.squeeze()])
        
        # Update ontology
        self.ontology.update_ontology([combined_features])
        
        return combined_features
    
    def incorporate_feedback(self, item_features, feedback):
        """Incorporate user feedback for continuous learning."""
        # Update feature clusters based on feedback
        if feedback['is_correct']:
            cluster_id = feedback['cluster_id']
            # Update cluster centroid with new information
            current_centroid = self.ontology.feature_clusters[cluster_id]
            updated_centroid = (current_centroid + item_features) / 2
            self.ontology.feature_clusters[cluster_id] = updated_centroid
            
            logging.info(f"Updated cluster {cluster_id} based on feedback")
        else:
            # Create new cluster if classification was incorrect
            self.ontology.update_ontology([item_features])

    def evaluate_system(self, eval_data):
        """Evaluate system with safeguards"""
        try:
            # Convert eval_data to DataFrame if it's not already
            if not isinstance(eval_data, pd.DataFrame):
                eval_data = pd.DataFrame(eval_data)
                
            # Validate required columns
            missing_cols = [col for col in self.required_columns if col not in eval_data.columns]
            if missing_cols:
                logging.warning(f"Missing columns in evaluation data: {missing_cols}")
                # Add missing columns with default values
                for col in missing_cols:
                    eval_data[col] = None
                    
            # Process evaluation data
            results = []
            for _, item in eval_data.iterrows():
                try:
                    prediction = self.process_item(item)
                    if prediction:
                        results.append(prediction)
                except Exception as e:
                    logging.error(f"Error processing evaluation item: {e}")
                    continue
                    
            return {
                'total_processed': len(results),
                'success_rate': len(results) / len(eval_data) if len(eval_data) > 0 else 0,
                'results': results
            }
            
        except Exception as e:
            logging.error(f"Evaluation error: {e}")
            return None
    
    def generate_evaluation_report(self):
        """Generate a comprehensive evaluation report."""
        if not self.evaluation_results:
            logging.warning("No evaluation results available. Run evaluate_system() first.")
            return
        
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': self.evaluation_results['metrics'],
            'cluster_analysis': {
                'total_clusters': len(self.ontology.feature_clusters),
                'silhouette_score': self.evaluation_results['metrics']['silhouette_score']
            }
        }
        
        # Save visualizations
        if self.evaluation_results['visualizations']['feature_space']:
            self.evaluation_results['visualizations']['feature_space'].write_html(
                'feature_space_visualization.html'
            )
        
        if self.evaluation_results['visualizations']['cluster_evolution']:
            self.evaluation_results['visualizations']['cluster_evolution'].write_html(
                'cluster_evolution.html'
            )
        
        if self.evaluation_results['visualizations']['confusion_matrix']:
            self.evaluation_results['visualizations']['confusion_matrix'].savefig(
                'confusion_matrix.png'
            )
        
        # Save report to JSON
        with open('evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        logging.info("Evaluation report generated successfully")
        return report

class EvaluationMetrics:
    """Handles evaluation and visualization of system performance."""

    def __init__(self):
        self.metrics_history = defaultdict(list)

    def calculate_cluster_metrics(self, features, cluster_labels):
        """Calculate clustering quality metrics."""
        try:
            silhouette = silhouette_score(features, cluster_labels)
            return {
                'silhouette_score': silhouette,
                'n_clusters': len(set(cluster_labels))
            }
        except Exception as e:
            logging.error(f"Error calculating cluster metrics: {str(e)}")
            return None

    def visualize_feature_space(self, features, labels, title="Feature Space Visualization"):
        """Create interactive visualization of the feature space using t-SNE."""
        try:
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            tsne = TSNE(n_components=2, random_state=42)
            features_2d = tsne.fit_transform(features_scaled)

            fig = px.scatter(
                x=features_2d[:, 0],
                y=features_2d[:, 1],
                color=labels,
                title=title,
                labels={'color': 'Cluster'},
                template='plotly_white'
            )
            fig.update_traces(marker=dict(size=8))
            return fig
        except Exception as e:
            logging.error(f"Error creating feature space visualization: {str(e)}")
            return None

    def plot_cluster_evolution(self):
        """Visualize how clusters have evolved over time."""
        try:
            history = pd.DataFrame(self.metrics_history)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=history.index,
                y=history['n_clusters'],
                name='Number of Clusters',
                mode='lines+markers'
            ))

            fig.add_trace(go.Scatter(
                x=history.index,
                y=history['silhouette_score'],
                name='Silhouette Score',
                mode='lines+markers',
                yaxis='y2'
            ))

            fig.update_layout(
                title='Cluster Evolution Over Time',
                xaxis_title='Time',
                yaxis_title='Number of Clusters',
                yaxis2=dict(
                    title='Silhouette Score',
                    overlaying='y',
                    side='right'
                ),
                template='plotly_white'
            )

            return fig
        except Exception as e:
            logging.error(f"Error creating cluster evolution plot: {str(e)}")
            return None

    def create_confusion_matrix(self, true_labels, predicted_labels, categories):
        """Create and visualize confusion matrix for classification results."""
        try:
            cm = confusion_matrix(true_labels, predicted_labels)
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=categories,
                yticklabels=categories
            )
            plt.title('Classification Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
            return plt.gcf()
        except Exception as e:
            logging.error(f"Error creating confusion matrix: {str(e)}")
            return None
        
class DynamicInputProcessor:
    """Processor for handling dynamic fashion item inputs."""

    def __init__(self, system):
        self.system = system
        self.supported_image_types = {'.jpg', '.jpeg', '.png', '.webp'}
        self.processing_history = []

    def process_fashion_item(self, image_input, description, metadata=None):
        """Process a single fashion item."""
        try:
            image_url = self._validate_and_process_image(image_input)
            if not image_url:
                raise ValueError("Invalid or inaccessible image")

            cleaned_description = self._validate_and_clean_description(description)
            if not cleaned_description:
                raise ValueError("Invalid description")

            features = self.system.process_new_item(image_url, cleaned_description)
            if features is None:
                raise ValueError("Feature extraction failed")

            cluster_assignment = self._assign_to_cluster(features)

            if cluster_assignment['confidence'] < 0.3:
                new_cluster_id = self._create_new_trend(features)
                cluster_assignment = {
                    'cluster_id': new_cluster_id,
                    'confidence': 1.0,
                    'is_new_trend': True
                }

            result = {
                'cluster': cluster_assignment,
                'features': features.tolist() if features is not None else [],
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {},
                'processing_details': {
                    'image_url': image_url or '',
                    'description': cleaned_description or ''
                },
                'status': 'processed'
            }

            self.processing_history.append(result)
            return result
        except Exception as e:
            logging.error(f"Error processing fashion item: {str(e)}")
            return {'error': str(e)}

    def _validate_and_process_image(self, image_input):
        """Validate and process image input"""
        try:
            if isinstance(image_input, str):  # URL or path
                if image_input.startswith('http'):
                    response = requests.get(image_input)
                    img = Image.open(BytesIO(response.content))
                else:
                    img = Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                img = image_input
            else:
                raise ValueError("Invalid image input format")
            
            # Validate image
            if img.size[0] < 100 or img.size[1] < 100:
                raise ValueError("Image too small")
                
            # Upload to S3
            img_path = f"images/{uuid.uuid4()}.jpg"
            img.save(img_path, "JPEG")
            self.s3_client.upload_file(img_path, self.bucket_name, img_path)
            os.remove(img_path)  # Clean up
            
            return f"s3://{self.bucket_name}/{img_path}"
        except Exception as e:
            logging.error(f"Image processing error: {e}")
            return None

    def _validate_and_clean_description(self, description):
        """Validate and clean fashion item description"""
        try:
            if not description or not isinstance(description, str):
                return None
            
            # Remove special characters and extra whitespace
            cleaned = re.sub(r'[^\w\s]', '', description)
            cleaned = ' '.join(cleaned.split())
            
            # Basic validation
            if len(cleaned) < 10:
                logging.warning("Description too short")
                return None
                
            return cleaned.lower()
        except Exception as e:
            logging.error(f"Error cleaning description: {e}")
            return None

    def _assign_to_cluster(self, features):
        """Assign features to the closest cluster and calculate confidence."""
        distances = {
            cluster_id: np.linalg.norm(features - centroid)
            for cluster_id, centroid in self.system.ontology.feature_clusters.items()
        }

        if not distances:
            return {'cluster_id': 0, 'confidence': 1.0, 'is_new_trend': True}

        closest_cluster = min(distances.items(), key=lambda x: x[1])
        max_distance = max(distances.values())

        return {
            'cluster_id': closest_cluster[0],
            'confidence': 1 - (closest_cluster[1] / max_distance),
            'is_new_trend': False
        }

    def _create_new_trend(self, features):
        """Create a new trend cluster and return its ID."""
        new_cluster_id = len(self.system.ontology.feature_clusters)
        self.system.ontology.feature_clusters[new_cluster_id] = features
        self.system.ontology.graph.add_node(
            f"cluster_{new_cluster_id}",
            features=features,
            creation_date=datetime.now()
        )
        logging.info(f"Created new trend cluster: {new_cluster_id}")
        return new_cluster_id

class FashionSystem:
    def __init__(self, data_directory=None, image_dir='fashion_images'):
        self.required_columns = {
            'product_id': str,
            'name': str, 
            'brand': str,
            'price': float,
            'category': str,
            'style_attributes': dict,
            'features': list,
            'feature_image_s3': str,
            'description': str
        }
        self.image_dir = os.path.join(os.getcwd(), image_dir)
        self.data_directory = data_directory
        self.df = None
        
    def _validate_column_types(self, df):
        """Validate column data types"""
        for col, dtype in self.required_columns.items():
            if col not in df.columns:
                df[col] = None
            else:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    logging.warning(f"Error converting {col} to {dtype}: {e}")
        return df
    
    def load_data(self):
        """Load and validate data"""
        try:
            if not self.data_directory:
                raise ValueError("Data directory not specified")
                
            df = pd.read_csv(os.path.join(self.data_directory, 'fashion_data.csv'))
            self.df = self._validate_column_types(df)
            
            logging.info(f"Loaded data with {len(self.df)} rows")
            return self.df
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def _validate_and_process_image(self, image_path):
        """Process local image instead of URL"""
        try:
            if os.path.exists(image_path):
                return Image.open(image_path)
            return None
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {str(e)}")
            return None

    def train(self):
        """Train model with validated data"""
        try:
            logging.info("Starting training process...")
            
            # Validate description column
            if 'description' not in self.df.columns:
                raise ValueError("Missing required column: description")
                
            # Process batches
            for batch in self._get_batches(self.df):
                if 'description' not in batch:
                    logging.error("Missing description in batch")
                    continue
                    
                self._process_batch(batch)
                
        except Exception as e:
            logging.error(f"Training error: {e}")
            raise

    def evaluate_system(self, eval_data):
        """Evaluate system with safeguards"""
        try:
            # Ensure data has required columns
            if 'feature_image_s3' not in eval_data.columns:
                self._ensure_required_columns()
            
            # Calculate metrics
            metrics = self.evaluator.calculate_cluster_metrics(
                eval_data, eval_data['category']
            )
            self.evaluator.metrics_history['silhouette_score'].append(
                metrics['silhouette_score']
            )
            self.evaluator.metrics_history['n_clusters'].append(
                metrics['n_clusters']
            )
            
            # Generate visualizations
            feature_space_viz = self.evaluator.visualize_feature_space(
                eval_data, eval_data['category']
            )
            cluster_evolution = self.evaluator.plot_cluster_evolution()
            confusion_mat = self.evaluator.create_confusion_matrix(
                eval_data['category'], 
                eval_data['category'],
                sorted(set(eval_data['category']))
            )
            
            # Store evaluation results
            self.evaluation_results = {
                'metrics': metrics,
                'visualizations': {
                    'feature_space': feature_space_viz,
                    'cluster_evolution': cluster_evolution,
                    'confusion_matrix': confusion_mat
                }
            }
            
            logging.info("System evaluation completed")
            return self.evaluation_results
        except Exception as e:
            logging.error(f"Evaluation error: {e}")
            raise

    def process_fashion_item(self, image_input, description, metadata=None):
        try:
            # Validate data schema first
            self.df = self._validate_data_schema(self.df)
            
            # Process image and get S3 URL
            image_url = self._validate_and_process_image(image_input)
            if image_url:
                self.df.at[len(self.df), 'feature_image_s3'] = image_url
            # Process description and get cleaned text
            cleaned_description = self._validate_and_clean_description(description)
            if cleaned_description:
                self.df.at[len(self.df), 'description'] = cleaned_description

            # Process metadata
            if metadata:
                self.df.at[len(self.df), 'metadata'] = metadata

            # Process the item using the system
            result = self.system.process_new_item(image_url, cleaned_description)
            return result
        except Exception as e:
            logging.error(f"Error processing fashion item: {str(e)}")
            raise

def create_gradio_interface(system):
    """Create an enhanced Gradio interface."""
    input_processor = DynamicInputProcessor(system)

    def process_and_visualize(image, description, category=None, brand=None):
        metadata = {
            'category': category,
            'brand': brand,
            'submission_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        results = input_processor.process_fashion_item(image, description, metadata)

        if 'error' in results:
            return f"Error: {results['error']}"

        output = f"""
        Processing Results:
        ==================
        Cluster Assignment: {results['cluster_id']}
        Confidence: {results['confidence']:.2%}
        {'[NEW TREND DETECTED]' if results.get('is_new_trend') else ''}

        Item Details:
        -------------
        Category: {category or 'Not specified'}
        Brand: {brand or 'Not specified'}
        Processed at: {results['timestamp']}

        System Status:
        -------------
        Total Clusters: {len(system.ontology.feature_clusters)}
        Total Items Processed: {len(input_processor.processing_history)}
        """

def main():
    """
    Main function orchestrating the complete Fashion Intelligence System workflow,
    utilizing existing components and providing comprehensive evaluation.
    """
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('fashion_system.log'),
                logging.StreamHandler()
            ]
        )
        logging.info("Starting Fashion Intelligence System")

        # Create output directories
        output_dir = "fashion_system_output"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'evaluations'), exist_ok=True)

        # Initialize system
        data_dir = r"NXT Hackathon Data"
        logging.info(f"Initializing system with data from: {data_dir}")
        system = FashionIntelligenceSystem(data_dir)

        # Training phase using existing train method
        logging.info("Starting training phase...")
        system.train()
        logging.info("Training completed successfully")

        # Evaluation phase using existing evaluation methods
        logging.info("Starting system evaluation...")
        
        # Get evaluation data
        full_data = system.preprocessor.process_all()
        eval_data = full_data.sample(n=min(1000, len(full_data)), random_state=42)
        logging.info(f"Selected {len(eval_data)} samples for evaluation")

        # Run comprehensive evaluation
        evaluation_results = system.evaluate_system(eval_data)
        
        # Generate evaluation report using existing method
        report = system.generate_evaluation_report()
        
        # Save all evaluation artifacts
        logging.info("Saving evaluation results...")
        
        # Save visualizations using existing methods from EvaluationMetrics
        viz_dir = os.path.join(output_dir, 'visualizations')
        
        if evaluation_results['visualizations']['feature_space']:
            feature_space_path = os.path.join(viz_dir, 'feature_space_visualization.html')
            evaluation_results['visualizations']['feature_space'].write_html(feature_space_path)
            logging.info(f"Feature space visualization saved to {feature_space_path}")

        if evaluation_results['visualizations']['cluster_evolution']:
            cluster_evolution_path = os.path.join(viz_dir, 'cluster_evolution.html')
            evaluation_results['visualizations']['cluster_evolution'].write_html(cluster_evolution_path)
            logging.info(f"Cluster evolution plot saved to {cluster_evolution_path}")

        if evaluation_results['visualizations']['confusion_matrix']:
            confusion_matrix_path = os.path.join(viz_dir, 'confusion_matrix.png')
            evaluation_results['visualizations']['confusion_matrix'].savefig(confusion_matrix_path)
            logging.info(f"Confusion matrix saved to {confusion_matrix_path}")

        # Print comprehensive metrics summary
        print("\nSystem Performance Summary:")
        print("-" * 50)
        print(f"Number of Clusters: {report['cluster_analysis']['total_clusters']}")
        print(f"Silhouette Score: {report['cluster_analysis']['silhouette_score']:.4f}")
        print(f"Evaluation Timestamp: {report['timestamp']}")
        print("-" * 50)

        # Create and launch Gradio interface using the existing function
        logging.info("Launching interactive interface...")
        iface = create_gradio_interface(system)
        iface.launch(share=True)
        
    except Exception as e:
        logging.error(f"Error in main workflow: {str(e)}")
        raise

if __name__ == "__main__":
    main()