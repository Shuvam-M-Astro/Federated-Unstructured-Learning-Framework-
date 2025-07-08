"""
Generate sample data for federated learning framework.

This script creates various types of sample data including text, images, tabular data,
PDFs, and web scraping examples for testing the federated learning system.
"""

import os
import json
import logging
import argparse
import random
import string
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# PDF generation
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    PDF_GENERATION_AVAILABLE = True
except ImportError:
    PDF_GENERATION_AVAILABLE = False
    logging.warning("PDF generation not available. Install reportlab")

# HTML generation
try:
    from jinja2 import Template
    HTML_GENERATION_AVAILABLE = True
except ImportError:
    HTML_GENERATION_AVAILABLE = False
    logging.warning("HTML generation not available. Install jinja2")

logger = logging.getLogger(__name__)


class SampleDataGenerator:
    """Generate sample data for federated learning."""
    
    def __init__(self, output_dir: str = "data"):
        """Initialize data generator.
        
        Args:
            output_dir: Output directory for generated data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample text content for generation
        self.sample_texts = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
            "Federated learning enables training machine learning models across decentralized data sources.",
            "Deep learning uses neural networks with multiple layers to learn complex patterns in data.",
            "Natural language processing helps computers understand and generate human language.",
            "Computer vision enables machines to interpret and understand visual information.",
            "Data science combines statistics, programming, and domain expertise to extract insights.",
            "Big data refers to large, complex datasets that require specialized tools for analysis.",
            "Cloud computing provides on-demand access to computing resources over the internet.",
            "Cybersecurity protects digital systems from unauthorized access and attacks.",
            "Blockchain technology creates secure, decentralized ledgers for digital transactions."
        ]
        
        # Sample topics for document generation
        self.topics = [
            "artificial_intelligence", "machine_learning", "data_science", 
            "cybersecurity", "cloud_computing", "blockchain", "iot", 
            "robotics", "quantum_computing", "edge_computing"
        ]
    
    def generate_text_data(self, num_files: int = 100, num_nodes: int = 3):
        """Generate sample text data.
        
        Args:
            num_files: Number of text files to generate
            num_nodes: Number of federated nodes
        """
        logger.info(f"Generating {num_files} text files across {num_nodes} nodes")
        
        for node_id in range(num_nodes):
            node_dir = self.output_dir / f"node{node_id + 1}" / "text"
            node_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(num_files):
                # Generate random text content
                num_sentences = random.randint(3, 8)
                content = []
                
                for _ in range(num_sentences):
                    sentence = random.choice(self.sample_texts)
                    # Add some variation
                    if random.random() > 0.5:
                        sentence += f" This is additional information about {random.choice(self.topics)}."
                    content.append(sentence)
                
                text_content = " ".join(content)
                
                # Save as text file
                text_path = node_dir / f"document_{i:03d}.txt"
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                
                # Save as JSON with metadata
                json_path = node_dir / f"document_{i:03d}.json"
                json_data = {
                    'content': text_content,
                    'metadata': {
                        'node_id': node_id + 1,
                        'file_id': i,
                        'topic': random.choice(self.topics),
                        'length': len(text_content),
                        'sentences': num_sentences
                    }
                }
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2)
    
    def generate_image_data(self, num_files: int = 50, num_nodes: int = 3):
        """Generate sample image data.
        
        Args:
            num_files: Number of images to generate
            num_nodes: Number of federated nodes
        """
        logger.info(f"Generating {num_files} images across {num_nodes} nodes")
        
        for node_id in range(num_nodes):
            node_dir = self.output_dir / f"node{node_id + 1}" / "images"
            node_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(num_files):
                # Create a simple image with text
                img_size = (400, 300)
                img = Image.new('RGB', img_size, color='white')
                draw = ImageDraw.Draw(img)
                
                # Add some geometric shapes
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                draw.rectangle([50, 50, 350, 250], outline=color, width=3)
                draw.ellipse([100, 100, 300, 200], fill=color)
                
                # Add text
                text = f"Image {i} from Node {node_id + 1}"
                draw.text((150, 150), text, fill='black')
                
                # Save as PNG
                png_path = node_dir / f"image_{i:03d}.png"
                img.save(png_path)
                
                # Save as JPG
                jpg_path = node_dir / f"image_{i:03d}.jpg"
                img.save(jpg_path, 'JPEG', quality=85)
    
    def generate_tabular_data(self, num_rows: int = 500, num_nodes: int = 3):
        """Generate sample tabular data.
        
        Args:
            num_rows: Number of rows to generate
            num_nodes: Number of federated nodes
        """
        logger.info(f"Generating {num_rows} rows of tabular data across {num_nodes} nodes")
        
        for node_id in range(num_nodes):
            node_dir = self.output_dir / f"node{node_id + 1}" / "tabular"
            node_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate synthetic data
            np.random.seed(42 + node_id)  # Different seed for each node
            
            age = np.random.normal(35, 10, num_rows).astype(int)
            age = np.clip(age, 18, 80)
            
            income = np.random.lognormal(10.5, 0.5, num_rows).astype(int)
            income = np.clip(income, 20000, 200000)
            
            education_years = np.random.normal(14, 3, num_rows).astype(int)
            education_years = np.clip(education_years, 8, 22)
            
            credit_score = np.random.normal(700, 100, num_rows).astype(int)
            credit_score = np.clip(credit_score, 300, 850)
            
            # Create approval based on features
            approval = ((age > 25) & (income > 50000) & (credit_score > 650)).astype(int)
            
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
    
    def generate_pdf_data(self, num_files: int = 20, num_nodes: int = 3):
        """Generate sample PDF documents.
        
        Args:
            num_files: Number of PDF files to generate
            num_nodes: Number of federated nodes
        """
        if not PDF_GENERATION_AVAILABLE:
            logger.warning("PDF generation not available. Skipping PDF generation.")
            return
        
        logger.info(f"Generating {num_files} PDF files across {num_nodes} nodes")
        
        for node_id in range(num_nodes):
            node_dir = self.output_dir / f"node{node_id + 1}" / "pdfs"
            node_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(num_files):
                pdf_path = node_dir / f"document_{i:03d}.pdf"
                
                # Create PDF with reportlab
                doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
                story = []
                
                # Add title
                styles = getSampleStyleSheet()
                title = Paragraph(f"Sample Document {i} from Node {node_id + 1}", styles['Title'])
                story.append(title)
                
                # Add content
                topic = random.choice(self.topics)
                content = f"""
                This is a sample document about {topic.replace('_', ' ')}.
                
                {random.choice(self.sample_texts)}
                
                Key points:
                • Point 1: Important information about {topic}
                • Point 2: Additional details and insights
                • Point 3: Further analysis and conclusions
                
                This document was generated for federated learning testing purposes.
                """
                
                content_para = Paragraph(content, styles['Normal'])
                story.append(content_para)
                
                # Add a simple table
                table_data = [
                    ['Metric', 'Value', 'Unit'],
                    ['Accuracy', f'{random.uniform(0.8, 0.95):.3f}', '%'],
                    ['Precision', f'{random.uniform(0.75, 0.9):.3f}', '%'],
                    ['Recall', f'{random.uniform(0.7, 0.85):.3f}', '%'],
                    ['F1-Score', f'{random.uniform(0.75, 0.9):.3f}', '%']
                ]
                
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 1), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
                
                # Build PDF
                doc.build(story)
    
    def generate_web_data(self, num_files: int = 15, num_nodes: int = 3):
        """Generate sample HTML/web data.
        
        Args:
            num_files: Number of HTML files to generate
            num_nodes: Number of federated nodes
        """
        if not HTML_GENERATION_AVAILABLE:
            logger.warning("HTML generation not available. Skipping web data generation.")
            return
        
        logger.info(f"Generating {num_files} HTML files across {num_nodes} nodes")
        
        # HTML template
        html_template = Template("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .content { margin: 20px 0; }
        .table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        .table th, .table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .table th { background-color: #f2f2f2; }
        .footer { margin-top: 40px; padding: 20px; background-color: #f9f9f9; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Generated for Federated Learning Testing - Node {{ node_id }}</p>
    </div>
    
    <div class="content">
        <h2>About {{ topic }}</h2>
        <p>{{ content }}</p>
        
        <h3>Key Features</h3>
        <ul>
            {% for feature in features %}
            <li>{{ feature }}</li>
            {% endfor %}
        </ul>
        
        <h3>Performance Metrics</h3>
        <table class="table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {% for metric in metrics %}
                <tr>
                    <td>{{ metric.name }}</td>
                    <td>{{ metric.value }}</td>
                    <td>{{ metric.status }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        <h3>Related Links</h3>
        <ul>
            {% for link in links %}
            <li><a href="{{ link.url }}">{{ link.text }}</a></li>
            {% endfor %}
        </ul>
    </div>
    
    <div class="footer">
        <p>Document ID: {{ doc_id }} | Generated: {{ timestamp }}</p>
    </div>
</body>
</html>
        """)
        
        for node_id in range(num_nodes):
            node_dir = self.output_dir / f"node{node_id + 1}" / "web"
            node_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(num_files):
                topic = random.choice(self.topics)
                title = f"Sample Web Page {i} - {topic.replace('_', ' ').title()}"
                
                # Generate content
                content = random.choice(self.sample_texts)
                if random.random() > 0.5:
                    content += f" This page provides additional information about {topic.replace('_', ' ')}."
                
                # Generate features
                features = [
                    f"Feature 1: Advanced {topic} capabilities",
                    f"Feature 2: Scalable {topic} solutions",
                    f"Feature 3: Secure {topic} implementation",
                    f"Feature 4: High-performance {topic} processing"
                ]
                
                # Generate metrics
                metrics = [
                    {"name": "Accuracy", "value": f"{random.uniform(0.8, 0.95):.3f}%", "status": "Good"},
                    {"name": "Precision", "value": f"{random.uniform(0.75, 0.9):.3f}%", "status": "Excellent"},
                    {"name": "Recall", "value": f"{random.uniform(0.7, 0.85):.3f}%", "status": "Good"},
                    {"name": "F1-Score", "value": f"{random.uniform(0.75, 0.9):.3f}%", "status": "Excellent"}
                ]
                
                # Generate links
                links = [
                    {"url": f"https://example.com/{topic}/overview", "text": f"{topic.replace('_', ' ').title()} Overview"},
                    {"url": f"https://example.com/{topic}/tutorial", "text": f"{topic.replace('_', ' ').title()} Tutorial"},
                    {"url": f"https://example.com/{topic}/api", "text": f"{topic.replace('_', ' ').title()} API Documentation"}
                ]
                
                # Render HTML
                html_content = html_template.render(
                    title=title,
                    node_id=node_id + 1,
                    topic=topic,
                    content=content,
                    features=features,
                    metrics=metrics,
                    links=links,
                    doc_id=f"DOC_{node_id + 1}_{i:03d}",
                    timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                
                # Save HTML file
                html_path = node_dir / f"page_{i:03d}.html"
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                # Save as JSON with metadata
                json_path = node_dir / f"page_{i:03d}.json"
                json_data = {
                    'url': f"https://example.com/{topic}/page_{i}",
                    'title': title,
                    'content': content,
                    'metadata': {
                        'node_id': node_id + 1,
                        'file_id': i,
                        'topic': topic,
                        'features': features,
                        'metrics': metrics,
                        'links': links
                    }
                }
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2)
    
    def generate_mixed_data(self, num_files: int = 20, num_nodes: int = 3):
        """Generate mixed data types.
        
        Args:
            num_files: Number of files to generate
            num_nodes: Number of federated nodes
        """
        logger.info(f"Generating {num_files} mixed files across {num_nodes} nodes")
        
        for node_id in range(num_nodes):
            node_dir = self.output_dir / f"node{node_id + 1}" / "mixed"
            node_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (node_dir / "text").mkdir(exist_ok=True)
            (node_dir / "images").mkdir(exist_ok=True)
            (node_dir / "tabular").mkdir(exist_ok=True)
            
            for i in range(num_files):
                # Generate text file
                text_content = f"Mixed data sample {i} from node {node_id + 1}. {random.choice(self.sample_texts)}"
                text_path = node_dir / "text" / f"text_{i:03d}.txt"
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                
                # Generate simple image
                img = Image.new('RGB', (200, 150), color='lightblue')
                draw = ImageDraw.Draw(img)
                draw.text((50, 75), f"Image {i}", fill='black')
                img_path = node_dir / "images" / f"image_{i:03d}.png"
                img.save(img_path)
                
                # Generate tabular data
                df = pd.DataFrame({
                    'id': [i],
                    'value': [random.randint(1, 100)],
                    'category': [random.choice(['A', 'B', 'C'])],
                    'node_id': [node_id]
                })
                csv_path = node_dir / "tabular" / f"data_{i:03d}.csv"
                df.to_csv(csv_path, index=False)
    
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
        
        # Generate new data types
        self.generate_pdf_data(num_files=15, num_nodes=num_nodes)
        self.generate_web_data(num_files=10, num_nodes=num_nodes)
        
        # Create metadata file
        metadata = {
            'generated_at': pd.Timestamp.now().isoformat(),
            'num_nodes': num_nodes,
            'data_types': ['text', 'image', 'tabular', 'mixed', 'pdf', 'web'],
            'structure': {
                'text': '50 files per node',
                'image': '30 files per node',
                'tabular': '500 rows per node',
                'mixed': '10 files per node',
                'pdf': '15 files per node',
                'web': '10 files per node'
            },
            'capabilities': {
                'pdf_processing': PDF_GENERATION_AVAILABLE,
                'html_generation': HTML_GENERATION_AVAILABLE,
                'document_processing': True
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
                       choices=['text', 'image', 'tabular', 'mixed', 'pdf', 'web', 'all'],
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
            elif data_type == 'pdf':
                generator.generate_pdf_data(num_nodes=args.num_nodes)
            elif data_type == 'web':
                generator.generate_web_data(num_nodes=args.num_nodes)


if __name__ == "__main__":
    main() 