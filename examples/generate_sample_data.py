"""
Generate sample data for federated learning testing.
"""

import os
import json
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import string
from pathlib import Path
import argparse
import logging

logger = logging.getLogger(__name__)


class SampleDataGenerator:
    """Generate sample data for federated learning testing."""
    
    def __init__(self, output_dir: str = "data"):
        """Initialize data generator.
        
        Args:
            output_dir: Output directory for generated data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_text_data(self, num_files: int = 100, num_nodes: int = 3):
        """Generate sample text data.
        
        Args:
            num_files: Number of text files per node
            num_nodes: Number of federated nodes
        """
        logger.info(f"Generating text data for {num_nodes} nodes")
        
        # Sample text content
        sample_texts = [
            "This is a sample document for federated learning.",
            "Machine learning models can be trained collaboratively.",
            "Privacy-preserving techniques protect sensitive data.",
            "Distributed training enables large-scale model development.",
            "Federated learning allows training without data sharing.",
            "Differential privacy adds noise to protect individual records.",
            "Secure aggregation combines model updates safely.",
            "Multi-party computation enables secure computation.",
            "Homomorphic encryption allows computation on encrypted data.",
            "Zero-knowledge proofs verify computations without revealing data."
        ]
        
        for node_id in range(1, num_nodes + 1):
            node_dir = self.output_dir / f"node{node_id}" / "text"
            node_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(num_files):
                # Generate random text content
                num_sentences = random.randint(3, 8)
                content = []
                
                for _ in range(num_sentences):
                    base_text = random.choice(sample_texts)
                    # Add some variation
                    if random.random() > 0.5:
                        base_text += f" Additional information about {random.choice(['privacy', 'security', 'learning', 'data'])}."
                    content.append(base_text)
                
                # Add some random words
                random_words = [''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 8))) 
                              for _ in range(random.randint(2, 5))]
                content.extend(random_words)
                
                # Write to file
                file_path = node_dir / f"document_{i:03d}.txt"
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(' '.join(content))
                
                # Create some JSON files
                if i % 5 == 0:
                    json_data = {
                        'id': f"doc_{node_id}_{i}",
                        'content': ' '.join(content),
                        'metadata': {
                            'node_id': node_id,
                            'file_id': i,
                            'type': 'text',
                            'length': len(' '.join(content))
                        }
                    }
                    
                    json_path = node_dir / f"document_{i:03d}.json"
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2)
    
    def generate_image_data(self, num_files: int = 50, num_nodes: int = 3):
        """Generate sample image data.
        
        Args:
            num_files: Number of image files per node
            num_nodes: Number of federated nodes
        """
        logger.info(f"Generating image data for {num_nodes} nodes")
        
        # Create simple geometric shapes as images
        for node_id in range(1, num_nodes + 1):
            node_dir = self.output_dir / f"node{node_id}" / "images"
            node_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(num_files):
                # Create image with random shape
                img_size = (224, 224)
                img = Image.new('RGB', img_size, color='white')
                draw = ImageDraw.Draw(img)
                
                # Random shape type
                shape_type = random.choice(['circle', 'rectangle', 'triangle', 'line'])
                
                if shape_type == 'circle':
                    x1, y1 = random.randint(50, 174), random.randint(50, 174)
                    radius = random.randint(20, 50)
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    draw.ellipse([x1-radius, y1-radius, x1+radius, y1+radius], fill=color)
                
                elif shape_type == 'rectangle':
                    x1, y1 = random.randint(20, 100), random.randint(20, 100)
                    x2, y2 = random.randint(120, 200), random.randint(120, 200)
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    draw.rectangle([x1, y1, x2, y2], fill=color)
                
                elif shape_type == 'triangle':
                    points = [
                        (random.randint(50, 174), random.randint(50, 174)),
                        (random.randint(50, 174), random.randint(50, 174)),
                        (random.randint(50, 174), random.randint(50, 174))
                    ]
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    draw.polygon(points, fill=color)
                
                elif shape_type == 'line':
                    x1, y1 = random.randint(20, 200), random.randint(20, 200)
                    x2, y2 = random.randint(20, 200), random.randint(20, 200)
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    draw.line([x1, y1, x2, y2], fill=color, width=random.randint(2, 8))
                
                # Save image
                file_path = node_dir / f"image_{i:03d}.png"
                img.save(file_path)
                
                # Create some JPEG files
                if i % 3 == 0:
                    jpeg_path = node_dir / f"image_{i:03d}.jpg"
                    img.save(jpeg_path, 'JPEG', quality=random.randint(70, 95))
    
    def generate_tabular_data(self, num_rows: int = 1000, num_nodes: int = 3):
        """Generate sample tabular data.
        
        Args:
            num_rows: Number of rows per node
            num_nodes: Number of federated nodes
        """
        logger.info(f"Generating tabular data for {num_nodes} nodes")
        
        for node_id in range(1, num_nodes + 1):
            node_dir = self.output_dir / f"node{node_id}" / "tabular"
            node_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate synthetic data
            np.random.seed(node_id)  # Different seed for each node
            
            # Features
            age = np.random.normal(35, 10, num_rows).astype(int)
            age = np.clip(age, 18, 80)
            
            income = np.random.lognormal(10, 0.5, num_rows).astype(int)
            income = np.clip(income, 20000, 200000)
            
            education_years = np.random.normal(12, 3, num_rows).astype(int)
            education_years = np.clip(education_years, 8, 20)
            
            credit_score = np.random.normal(650, 100, num_rows).astype(int)
            credit_score = np.clip(credit_score, 300, 850)
            
            # Target variable (binary classification)
            # Simple rule: higher income and education -> higher probability of approval
            approval_prob = (income / 100000 + education_years / 20 + credit_score / 850) / 3
            approval = (np.random.random(num_rows) < approval_prob).astype(int)
            
            # Create DataFrame
            df = pd.DataFrame({
                'age': age,
                'income': income,
                'education_years': education_years,
                'credit_score': credit_score,
                'approval': approval,
                'node_id': node_id
            })
            
            # Save as CSV
            csv_path = node_dir / "data.csv"
            df.to_csv(csv_path, index=False)
            
            # Save as Excel
            excel_path = node_dir / "data.xlsx"
            df.to_excel(excel_path, index=False)
            
            # Save as JSON
            json_path = node_dir / "data.json"
            df.to_json(json_path, orient='records', indent=2)
            
            # Create some smaller files for testing
            for i in range(5):
                subset_df = df.sample(n=min(100, len(df)))
                subset_path = node_dir / f"subset_{i}.csv"
                subset_df.to_csv(subset_path, index=False)
    
    def generate_mixed_data(self, num_files: int = 20, num_nodes: int = 3):
        """Generate mixed data types.
        
        Args:
            num_files: Number of files per node
            num_nodes: Number of federated nodes
        """
        logger.info(f"Generating mixed data for {num_nodes} nodes")
        
        for node_id in range(1, num_nodes + 1):
            node_dir = self.output_dir / f"node{node_id}" / "mixed"
            node_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for different data types
            text_dir = node_dir / "text"
            image_dir = node_dir / "images"
            tabular_dir = node_dir / "tabular"
            
            text_dir.mkdir(exist_ok=True)
            image_dir.mkdir(exist_ok=True)
            tabular_dir.mkdir(exist_ok=True)
            
            # Generate mixed content
            for i in range(num_files):
                # Text content
                text_content = f"Sample text content for mixed data file {i} from node {node_id}."
                text_path = text_dir / f"text_{i:03d}.txt"
                with open(text_path, 'w') as f:
                    f.write(text_content)
                
                # Simple image
                img = Image.new('RGB', (100, 100), color=(random.randint(0, 255), 
                                                         random.randint(0, 255), 
                                                         random.randint(0, 255)))
                img_path = image_dir / f"image_{i:03d}.png"
                img.save(img_path)
                
                # Tabular data
                tabular_data = pd.DataFrame({
                    'feature1': np.random.randn(10),
                    'feature2': np.random.randn(10),
                    'label': np.random.randint(0, 2, 10)
                })
                tabular_path = tabular_dir / f"data_{i:03d}.csv"
                tabular_data.to_csv(tabular_path, index=False)
    
    def generate_all_data(self, num_nodes: int = 3):
        """Generate all types of sample data.
        
        Args:
            num_nodes: Number of federated nodes
        """
        logger.info("Generating all sample data types")
        
        # Generate different data types
        self.generate_text_data(num_files=50, num_nodes=num_nodes)
        self.generate_image_data(num_files=30, num_nodes=num_nodes)
        self.generate_tabular_data(num_rows=500, num_nodes=num_nodes)
        self.generate_mixed_data(num_files=10, num_nodes=num_nodes)
        
        # Create metadata file
        metadata = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'num_nodes': num_nodes,
            'data_types': ['text', 'image', 'tabular', 'mixed'],
            'structure': {
                'text': '50 files per node',
                'image': '30 files per node',
                'tabular': '500 rows per node',
                'mixed': '10 files per node'
            }
        }
        
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Sample data generated in {self.output_dir}")
        logger.info(f"Generated {num_nodes} nodes with various data types")


def main():
    """Main function to generate sample data."""
    parser = argparse.ArgumentParser(description='Generate sample data for federated learning')
    parser.add_argument('--output-dir', type=str, default='data', 
                       help='Output directory for generated data')
    parser.add_argument('--num-nodes', type=int, default=3, 
                       help='Number of federated nodes')
    parser.add_argument('--data-types', nargs='+', 
                       choices=['text', 'image', 'tabular', 'mixed', 'all'],
                       default=['all'], help='Types of data to generate')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create generator
    generator = SampleDataGenerator(args.output_dir)
    
    # Generate requested data types
    if 'all' in args.data_types:
        generator.generate_all_data(args.num_nodes)
    else:
        for data_type in args.data_types:
            if data_type == 'text':
                generator.generate_text_data(num_nodes=args.num_nodes)
            elif data_type == 'image':
                generator.generate_image_data(num_nodes=args.num_nodes)
            elif data_type == 'tabular':
                generator.generate_tabular_data(num_nodes=args.num_nodes)
            elif data_type == 'mixed':
                generator.generate_mixed_data(num_nodes=args.num_nodes)


if __name__ == "__main__":
    main() 