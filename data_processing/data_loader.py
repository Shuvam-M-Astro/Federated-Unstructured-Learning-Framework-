"""
Data loader for handling multiple unstructured data formats in federated learning.
"""

import os
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2

from utils.constants import (
    SUPPORTED_TEXT_EXTENSIONS, 
    SUPPORTED_IMAGE_EXTENSIONS, 
    SUPPORTED_TABULAR_EXTENSIONS,
    SUPPORTED_PDF_EXTENSIONS,
    SUPPORTED_WEB_EXTENSIONS,
    SUPPORTED_DOCUMENT_EXTENSIONS,
    DEFAULT_IMAGE_SIZE
)

# Import document processor
try:
    from data_processing.document_processor import DocumentProcessor, DocumentContent, WebPageContent
    DOCUMENT_PROCESSING_AVAILABLE = True
except ImportError:
    DOCUMENT_PROCESSING_AVAILABLE = False
    logging.warning("Document processing not available. Install required dependencies.")

logger = logging.getLogger(__name__)


class UnstructuredDataset(Dataset):
    """Dataset class for handling unstructured data."""
    
    def __init__(self, data_path: str, data_type: str, transform=None, target_column: Optional[str] = None):
        """Initialize dataset.
        
        Args:
            data_path: Path to data directory or file
            data_type: Type of data ('text', 'image', 'tabular', 'mixed', 'pdf', 'web', 'document')
            transform: Optional transform to apply
            target_column: For tabular data, column name for labels
        """
        self.data_path = Path(data_path)
        self.data_type = data_type
        self.transform = transform
        self.target_column = target_column
        
        # Initialize document processor if available
        self.document_processor = None
        if DOCUMENT_PROCESSING_AVAILABLE and data_type in ['pdf', 'web', 'document']:
            self.document_processor = DocumentProcessor()
        
        self.data = []
        self.labels = []
        self.metadata = []
        
        self._load_data()
    
    def _load_data(self):
        """Load data based on type."""
        if self.data_type == 'text':
            self._load_text_data()
        elif self.data_type == 'image':
            self._load_image_data()
        elif self.data_type == 'tabular':
            self._load_tabular_data()
        elif self.data_type == 'mixed':
            self._load_mixed_data()
        elif self.data_type == 'pdf':
            self._load_pdf_data()
        elif self.data_type == 'web':
            self._load_web_data()
        elif self.data_type == 'document':
            self._load_document_data()
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")
    
    def _load_pdf_data(self):
        """Load PDF data from files."""
        if not self.document_processor:
            raise ImportError("Document processing not available for PDF data")
        
        for file_path in self.data_path.rglob('*'):
            if file_path.suffix.lower() in SUPPORTED_PDF_EXTENSIONS:
                try:
                    # Process PDF using document processor
                    document_content = self.document_processor.process_document(str(file_path))
                    
                    # Extract label from filename or directory structure
                    label = self._extract_label_from_path(file_path)
                    
                    # Store text content as primary data
                    self.data.append(document_content.text)
                    self.labels.append(label)
                    self.metadata.append({
                        'file_path': str(file_path),
                        'file_size': file_path.stat().st_size,
                        'extraction_method': document_content.extraction_method,
                        'num_tables': len(document_content.tables),
                        'num_images': len(document_content.images),
                        'confidence': document_content.confidence,
                        'pdf_metadata': document_content.metadata
                    })
                    
                    # Store tables as additional data if available
                    if document_content.tables:
                        for i, table in enumerate(document_content.tables):
                            table_label = f"{label}_table_{i}"
                            self.data.append(table.to_dict('records'))
                            self.labels.append(table_label)
                            self.metadata.append({
                                'file_path': str(file_path),
                                'table_index': i,
                                'data_type': 'table',
                                'rows': len(table),
                                'columns': len(table.columns)
                            })
                    
                except Exception as e:
                    logger.warning(f"Failed to load PDF file {file_path}: {e}")
    
    def _load_web_data(self):
        """Load web scraped data."""
        if not self.document_processor:
            raise ImportError("Document processing not available for web data")
        
        # Check if data_path contains URLs or HTML files
        if self.data_path.is_file() and self.data_path.suffix.lower() in SUPPORTED_WEB_EXTENSIONS:
            # Single HTML file
            try:
                document_content = self.document_processor.process_document(str(self.data_path))
                
                self.data.append(document_content.text)
                self.labels.append(0)  # Default label
                self.metadata.append({
                    'file_path': str(self.data_path),
                    'extraction_method': document_content.extraction_method,
                    'data_type': 'web_page'
                })
                
            except Exception as e:
                logger.warning(f"Failed to load web file {self.data_path}: {e}")
        
        elif self.data_path.is_dir():
            # Directory containing URLs or HTML files
            for file_path in self.data_path.rglob('*'):
                if file_path.suffix.lower() in SUPPORTED_WEB_EXTENSIONS:
                    try:
                        document_content = self.document_processor.process_document(str(file_path))
                        
                        label = self._extract_label_from_path(file_path)
                        
                        self.data.append(document_content.text)
                        self.labels.append(label)
                        self.metadata.append({
                            'file_path': str(file_path),
                            'extraction_method': document_content.extraction_method,
                            'data_type': 'web_page',
                            'num_tables': len(document_content.tables),
                            'raw_html': document_content.raw_html is not None
                        })
                        
                    except Exception as e:
                        logger.warning(f"Failed to load web file {file_path}: {e}")
    
    def _load_document_data(self):
        """Load mixed document data (PDFs, HTML, etc.)."""
        if not self.document_processor:
            raise ImportError("Document processing not available for document data")
        
        for file_path in self.data_path.rglob('*'):
            if file_path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS:
                try:
                    # Process document using document processor
                    document_content = self.document_processor.process_document(str(file_path))
                    
                    # Extract label from filename or directory structure
                    label = self._extract_label_from_path(file_path)
                    
                    # Store text content as primary data
                    self.data.append(document_content.text)
                    self.labels.append(label)
                    self.metadata.append({
                        'file_path': str(file_path),
                        'file_size': file_path.stat().st_size,
                        'file_type': file_path.suffix.lower(),
                        'extraction_method': document_content.extraction_method,
                        'confidence': document_content.confidence,
                        'num_tables': len(document_content.tables),
                        'num_images': len(document_content.images)
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to load document file {file_path}: {e}")
    
    def _load_text_data(self):
        """Load text data from files."""
        for file_path in self.data_path.rglob('*'):
            if file_path.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract label from filename or directory structure
                    label = self._extract_label_from_path(file_path)
                    
                    self.data.append(content)
                    self.labels.append(label)
                    self.metadata.append({
                        'file_path': str(file_path),
                        'file_size': file_path.stat().st_size,
                        'encoding': 'utf-8'
                    })
                except Exception as e:
                    logger.warning(f"Failed to load text file {file_path}: {e}")
    
    def _load_image_data(self):
        """Load image data from files."""
        for file_path in self.data_path.rglob('*'):
            if file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                try:
                    image = Image.open(file_path).convert('RGB')
                    
                    # Extract label from filename or directory structure
                    label = self._extract_label_from_path(file_path)
                    
                    self.data.append(image)
                    self.labels.append(label)
                    self.metadata.append({
                        'file_path': str(file_path),
                        'image_size': image.size,
                        'mode': image.mode
                    })
                except Exception as e:
                    logger.warning(f"Failed to load image file {file_path}: {e}")
    
    def _load_tabular_data(self):
        """Load tabular data from files."""
        for file_path in self.data_path.rglob('*'):
            if file_path.suffix.lower() in SUPPORTED_TABULAR_EXTENSIONS:
                try:
                    if file_path.suffix.lower() == '.csv':
                        df = pd.read_csv(file_path)
                    elif file_path.suffix.lower() in {'.xlsx', '.xls'}:
                        df = pd.read_excel(file_path)
                    elif file_path.suffix.lower() == '.parquet':
                        df = pd.read_parquet(file_path)
                    elif file_path.suffix.lower() == '.json':
                        df = pd.read_json(file_path)
                    
                    # Extract features and labels
                    if self.target_column and self.target_column in df.columns:
                        features = df.drop(columns=[self.target_column])
                        labels = df[self.target_column]
                    else:
                        features = df
                        labels = pd.Series([0] * len(df))  # Default label
                    
                    self.data.extend(features.values.tolist())
                    # Handle labels properly - convert to list if it's a pandas Series
                    if hasattr(labels, 'tolist'):
                        self.labels.extend(labels.tolist())
                    else:
                        self.labels.extend(list(labels))
                    
                    for i in range(len(df)):
                        self.metadata.append({
                            'file_path': str(file_path),
                            'row_index': i,
                            'columns': list(df.columns)
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to load tabular file {file_path}: {e}")
    
    def _load_mixed_data(self):
        """Load mixed data types."""
        # Load all supported data types
        self._load_text_data()
        self._load_image_data()
        self._load_tabular_data()
        
        # Also load document data if available
        if DOCUMENT_PROCESSING_AVAILABLE:
            self._load_document_data()
    
    def _extract_label_from_path(self, file_path: Path) -> int:
        """Extract label from file path or directory structure.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted label (integer)
        """
        # Try to extract label from directory name
        for parent in file_path.parents:
            if parent.name.isdigit():
                return int(parent.name)
        
        # Try to extract from filename
        filename = file_path.stem
        if filename.isdigit():
            return int(filename)
        
        # Default label
        return 0
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Any, int, Dict]:
        """Get item from dataset.
        
        Args:
            idx: Index of item
            
        Returns:
            Tuple of (data, label, metadata)
        """
        data = self.data[idx]
        label = self.labels[idx]
        metadata = self.metadata[idx]
        
        # Apply transform if available
        if self.transform:
            data = self.transform(data)
        
        return data, label, metadata


class DataProcessor:
    """Process and prepare data for federated learning."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize data processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.supported_formats = config.get('supported_formats', ['text', 'image', 'tabular', 'pdf', 'web', 'document'])
        self.max_file_size = config.get('max_file_size', 100) * 1024 * 1024  # Convert to bytes
        
        # Initialize document processor if needed
        self.document_processor = None
        if DOCUMENT_PROCESSING_AVAILABLE and any(fmt in ['pdf', 'web', 'document'] for fmt in self.supported_formats):
            self.document_processor = DocumentProcessor(config.get('document_processing', {}))
        
        # Initialize transforms
        self.transforms = self._initialize_transforms()
    
    def _initialize_transforms(self) -> Dict[str, Any]:
        """Initialize data transforms for different data types.
        
        Returns:
            Dictionary of transforms
        """
        transforms_dict = {}
        
        # Image transforms
        if 'image' in self.supported_formats:
            transforms_dict['image'] = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        
        # Text transforms (placeholder for tokenization)
        if 'text' in self.supported_formats:
            transforms_dict['text'] = lambda x: x  # Identity transform for now
        
        # Tabular transforms
        if 'tabular' in self.supported_formats:
            transforms_dict['tabular'] = lambda x: torch.tensor(x, dtype=torch.float32)
        
        # Document transforms
        if 'document' in self.supported_formats:
            transforms_dict['document'] = lambda x: x  # Identity transform for now
        
        # PDF transforms
        if 'pdf' in self.supported_formats:
            transforms_dict['pdf'] = lambda x: x  # Identity transform for now
        
        # Web transforms
        if 'web' in self.supported_formats:
            transforms_dict['web'] = lambda x: x  # Identity transform for now
        
        return transforms_dict
    
    def create_dataloader(self, data_path: str, data_type: str, 
                         batch_size: int = 32, shuffle: bool = True,
                         target_column: Optional[str] = None) -> DataLoader:
        """Create PyTorch DataLoader for the dataset.
        
        Args:
            data_path: Path to data
            data_type: Type of data
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            target_column: For tabular data, column name for labels
            
        Returns:
            PyTorch DataLoader
        """
        if data_type not in self.supported_formats:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        transform = self.transforms.get(data_type)
        
        dataset = UnstructuredDataset(
            data_path=data_path,
            data_type=data_type,
            transform=transform,
            target_column=target_column
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Set to 0 for federated learning
            drop_last=True
        )
    
    def get_data_info(self, data_path: str, data_type: str) -> Dict[str, Any]:
        """Get information about the dataset.
        
        Args:
            data_path: Path to data
            data_type: Type of data
            
        Returns:
            Dictionary with dataset information
        """
        dataset = UnstructuredDataset(data_path, data_type)
        
        info = {
            'total_samples': len(dataset),
            'data_type': data_type,
            'labels': list(set(dataset.labels)),
            'num_classes': len(set(dataset.labels)),
            'metadata': dataset.metadata[:10]  # First 10 samples
        }
        
        # Add document processing info if available
        if self.document_processor:
            info['document_processing'] = self.document_processor.get_available_processors()
        
        return info
    
    def validate_data(self, data_path: str, data_type: str) -> bool:
        """Validate data at the given path.
        
        Args:
            data_path: Path to data directory
            data_type: Type of data ('text', 'image', 'tabular', 'mixed', 'pdf', 'web', 'document')
            
        Returns:
            True if data is valid
        """
        if not os.path.exists(data_path):
            logger.error(f"Data path does not exist: {data_path}")
            return False
        
        if data_type not in ['text', 'image', 'tabular', 'mixed', 'pdf', 'web', 'document']:
            logger.error(f"Unsupported data type: {data_type}")
            return False
        
        # Check for minimum data requirements
        if data_type == 'text':
            return self._validate_text_data(data_path)
        elif data_type == 'image':
            return self._validate_image_data(data_path)
        elif data_type == 'tabular':
            return self._validate_tabular_data(data_path)
        elif data_type == 'mixed':
            return self._validate_mixed_data(data_path)
        elif data_type == 'pdf':
            return self._validate_pdf_data(data_path)
        elif data_type == 'web':
            return self._validate_web_data(data_path)
        elif data_type == 'document':
            return self._validate_document_data(data_path)
        
        return False
    
    def _validate_pdf_data(self, data_path: str) -> bool:
        """Validate PDF data files."""
        if not DOCUMENT_PROCESSING_AVAILABLE:
            logger.error("Document processing not available for PDF validation")
            return False
        
        pdf_files = self._get_files_by_extension(data_path, SUPPORTED_PDF_EXTENSIONS)
        if len(pdf_files) < 1:  # Minimum 1 PDF file
            logger.warning(f"Insufficient PDF files: {len(pdf_files)}")
            return False
        return True
    
    def _validate_web_data(self, data_path: str) -> bool:
        """Validate web data files."""
        if not DOCUMENT_PROCESSING_AVAILABLE:
            logger.error("Document processing not available for web data validation")
            return False
        
        web_files = self._get_files_by_extension(data_path, SUPPORTED_WEB_EXTENSIONS)
        if len(web_files) < 1:  # Minimum 1 web file
            logger.warning(f"Insufficient web files: {len(web_files)}")
            return False
        return True
    
    def _validate_document_data(self, data_path: str) -> bool:
        """Validate document data files."""
        if not DOCUMENT_PROCESSING_AVAILABLE:
            logger.error("Document processing not available for document validation")
            return False
        
        document_files = self._get_files_by_extension(data_path, SUPPORTED_DOCUMENT_EXTENSIONS)
        if len(document_files) < 1:  # Minimum 1 document file
            logger.warning(f"Insufficient document files: {len(document_files)}")
            return False
        return True
    
    def _validate_text_data(self, data_path: str) -> bool:
        """Validate text data files."""
        text_files = self._get_files_by_extension(data_path, SUPPORTED_TEXT_EXTENSIONS)
        if len(text_files) < 10:  # Minimum 10 text files
            logger.warning(f"Insufficient text files: {len(text_files)}")
            return False
        return True
    
    def _validate_image_data(self, data_path: str) -> bool:
        """Validate image data files."""
        image_files = self._get_files_by_extension(data_path, SUPPORTED_IMAGE_EXTENSIONS)
        if len(image_files) < 10:  # Minimum 10 image files
            logger.warning(f"Insufficient image files: {len(image_files)}")
            return False
        return True
    
    def _validate_tabular_data(self, data_path: str) -> bool:
        """Validate tabular data files."""
        tabular_files = self._get_files_by_extension(data_path, SUPPORTED_TABULAR_EXTENSIONS)
        if len(tabular_files) == 0:
            logger.error("No tabular data files found")
            return False
        return True
    
    def _validate_mixed_data(self, data_path: str) -> bool:
        """Validate mixed data structure."""
        mixed_dir = os.path.join(data_path, 'mixed')
        if not os.path.exists(mixed_dir):
            logger.error("Mixed data directory not found")
            return False
        return True
    
    def _get_files_by_extension(self, data_path: str, extensions: set) -> List[str]:
        """Get files with specific extensions."""
        files = []
        for ext in extensions:
            files.extend(Path(data_path).rglob(f'*{ext}'))
        return [str(f) for f in files]


class TextProcessor:
    """Process text data for federated learning."""
    
    def __init__(self, max_length: int = 512, vocab_size: int = 10000):
        """Initialize text processor.
        
        Args:
            max_length: Maximum sequence length
            vocab_size: Vocabulary size
        """
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_built = False
    
    def build_vocabulary(self, texts: List[str]):
        """Build vocabulary from text data.
        
        Args:
            texts: List of text samples
        """
        from collections import Counter
        import re
        
        # Tokenize and count words
        word_counts = Counter()
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            word_counts.update(words)
        
        # Build vocabulary
        most_common = word_counts.most_common(self.vocab_size - 2)  # Reserve for PAD and UNK
        
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        
        for i, (word, _) in enumerate(most_common, start=2):
            self.word_to_idx[word] = i
            self.idx_to_word[i] = word
        
        self.vocab_built = True
        logger.info(f"Vocabulary built with {len(self.word_to_idx)} words")
    
    def text_to_sequence(self, text: str) -> List[int]:
        """Convert text to sequence of indices.
        
        Args:
            text: Input text
            
        Returns:
            List of word indices
        """
        import re
        
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")
        
        words = re.findall(r'\b\w+\b', text.lower())
        sequence = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
        
        # Pad or truncate to max_length
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence.extend([self.word_to_idx['<PAD>']] * (self.max_length - len(sequence)))
        
        return sequence
    
    def sequence_to_text(self, sequence: List[int]) -> str:
        """Convert sequence of indices back to text.
        
        Args:
            sequence: List of word indices
            
        Returns:
            Reconstructed text
        """
        words = [self.idx_to_word.get(idx, '<UNK>') for idx in sequence]
        return ' '.join(words).replace('<PAD>', '').strip() 