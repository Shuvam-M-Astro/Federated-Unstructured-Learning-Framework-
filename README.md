# Federated Learning for Unstructured Data

A comprehensive implementation of federated learning that can handle distributed, private datasets with different data formats while maintaining data privacy. Now with advanced document processing capabilities including PDF extraction and web scraping.

## 🚀 Features

- **Multi-format Data Support**: Handle text, images, tabular data, PDFs, web pages, and mixed formats
- **Advanced Document Processing**: PDF text extraction, table detection, image extraction, and OCR
- **Web Scraping**: Extract content from HTML pages with multiple scraping methods
- **Privacy-Preserving**: Differential privacy, secure aggregation, and local training
- **Distributed Architecture**: Scalable across multiple nodes
- **Heterogeneous Data**: Handle different data formats across federated nodes
- **Model Agnostic**: Support for various ML frameworks (PyTorch, TensorFlow)
- **Real-time Communication**: WebSocket-based client-server communication
- **Monitoring & Logging**: Comprehensive tracking of training progress

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Central       │    │   Federated     │    │   Federated     │
│   Server        │◄──►│   Node 1        │    │   Node N        │
│                 │    │                 │    │                 │
│ - Model         │    │ - Local Data    │    │ - Local Data    │
│ - Aggregation   │    │ - Training      │    │ - Training      │
│ - Coordination  │    │ - Privacy       │    │ - Privacy       │
│ - Monitoring    │    │ - Document      │    │ - Document      │
│                 │    │   Processing    │    │   Processing    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy, Pandas, Pillow
- FastAPI, WebSockets
- Cryptography, PySyft (for privacy)
- PDF Processing: PyPDF2, pdfplumber, pdf2image
- Web Scraping: trafilatura, newspaper3k, beautifulsoup4
- OCR: pytesseract, easyocr
- Docker (optional)

## 🛠️ Installation

```bash
# Clone the repository
git clone
cd federated-learning-unstructured

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install system dependencies for OCR
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr
# macOS:
brew install tesseract
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

## 🚀 Quick Start

### 1. Generate Sample Data (including PDFs and web pages)
```bash
# Generate all data types
python examples/generate_sample_data.py --data-types all

# Generate specific data types
python examples/generate_sample_data.py --data-types pdf web text image tabular
```

### 2. Start the Central Server
```bash
python server/central_server.py
```

### 3. Start Federated Nodes
```bash
# Terminal 1 - PDF data
python clients/federated_client.py --node-id 1 --data-path data/node1/pdfs --data-type pdf

# Terminal 2 - Web data
python clients/federated_client.py --node-id 2 --data-path data/node1/web --data-type web

# Terminal 3 - Mixed document data
python clients/federated_client.py --node-id 3 --data-path data/node1 --data-type document
```

### 4. Monitor Training
```bash
python monitoring/dashboard.py
```

### 5. Test Document Processing
```bash
# Run comprehensive tests
python test_document_processing.py
```

## 📁 Project Structure

```
federated-learning-unstructured/
├── server/                 # Central server implementation
├── clients/               # Federated client implementations
├── models/                # ML model definitions
├── data_processing/       # Data format handlers
│   ├── data_loader.py     # Main data loading logic
│   └── document_processor.py # PDF and web processing
├── privacy/              # Privacy-preserving mechanisms
├── communication/        # Network communication layer
├── monitoring/           # Training monitoring and visualization
├── utils/               # Utility functions
├── config/              # Configuration files
├── tests/               # Unit and integration tests
├── examples/            # Example datasets and usage
└── test_document_processing.py # Document processing tests
```

## 🔒 Privacy Features

- **Differential Privacy**: Add noise to gradients during training
- **Secure Aggregation**: Cryptographic aggregation of model updates
- **Local Training**: Data never leaves the client nodes
- **Model Encryption**: Encrypt model parameters during transmission

## 📊 Supported Data Formats

### Core Formats
- **Text**: Documents, emails, social media posts (.txt, .md, .json)
- **Images**: Photos, scanned documents, charts (.jpg, .png, .bmp, .tiff)
- **Tabular**: CSV, Excel, database exports (.csv, .xlsx, .parquet)

### Advanced Document Formats
- **PDF Documents**: Text extraction, table detection, image extraction, metadata (.pdf)
- **Web Pages**: HTML content extraction, link analysis, table parsing (.html, .htm)
- **Mixed Documents**: Combined processing of multiple formats
- **OCR Support**: Text extraction from scanned documents and images

### Processing Capabilities
- **PDF Processing**: 
  - Text extraction with layout preservation
  - Table detection and extraction (stream/lattice methods)
  - Image extraction and processing
  - Metadata extraction (author, creation date, etc.)
  - Multi-page document support

- **Web Scraping**:
  - Multiple scraping methods (trafilatura, newspaper3k, beautifulsoup)
  - Content cleaning and normalization
  - Link and image extraction
  - Table parsing from HTML
  - Rate limiting and retry mechanisms

- **OCR Processing**:
  - Multi-language support (English, Spanish, French, German, etc.)
  - Confidence scoring for extracted text
  - Image preprocessing for better accuracy
  - Support for various image formats

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_privacy.py
python -m pytest tests/test_communication.py

# Test document processing capabilities
python test_document_processing.py
```

## 📈 Performance

- **Scalability**: Tested with up to 100 federated nodes
- **Privacy**: ε-differential privacy with ε < 1.0
- **Accuracy**: Comparable to centralized training
- **Communication**: Efficient model compression
- **Document Processing**: Support for documents up to 50MB
- **Web Scraping**: Rate-limited with configurable delays

## 🔧 Configuration

### Document Processing Settings
```yaml
# config/config.yaml
document_processing:
  pdf:
    max_pages: 1000
    extraction_methods: ['text', 'tables', 'metadata']
    image_dpi: 300
  web_scraping:
    timeout: 30
    retry_attempts: 3
    delay: 1
    max_pages: 1000
  ocr:
    languages: ['en']
    confidence_threshold: 0.7
```

### Data Type Configuration
```yaml
data_processing:
  supported_formats: ['text', 'image', 'tabular', 'pdf', 'web', 'document']
  max_file_size: 100  # MB
```

## 📝 Usage Examples

### PDF Processing
```python
from data_processing.document_processor import PDFProcessor

# Initialize processor
pdf_processor = PDFProcessor(extraction_methods=['text', 'tables', 'metadata'])

# Process PDF
content = pdf_processor.process_pdf('document.pdf')
print(f"Text: {content.text[:200]}...")
print(f"Tables: {len(content.tables)}")
print(f"Metadata: {content.metadata}")
```

### Web Scraping
```python
from data_processing.document_processor import WebScraper

# Initialize scraper
scraper = WebScraper(timeout=30, delay=1)

# Scrape single URL
content = scraper.scrape_url('https://example.com', method='trafilatura')
print(f"Title: {content.title}")
print(f"Text: {content.text[:200]}...")
print(f"Links: {len(content.links)}")
```

### Federated Learning with Documents
```python
from data_processing.data_loader import UnstructuredDataset

# Load PDF data
pdf_dataset = UnstructuredDataset('data/node1/pdfs', 'pdf')
print(f"PDF dataset size: {len(pdf_dataset)}")

# Load web data
web_dataset = UnstructuredDataset('data/node1/web', 'web')
print(f"Web dataset size: {len(web_dataset)}")

# Load mixed document data
doc_dataset = UnstructuredDataset('data/node1', 'document')
print(f"Document dataset size: {len(doc_dataset)}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
