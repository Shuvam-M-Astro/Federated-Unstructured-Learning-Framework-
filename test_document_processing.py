#!/usr/bin/env python3
"""
Test script for document processing capabilities in federated learning framework.

This script demonstrates the new PDF processing and web scraping features.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from data_processing.document_processor import DocumentProcessor, PDFProcessor, WebScraper, OCRProcessor
from data_processing.data_loader import UnstructuredDataset, DataProcessor
from examples.generate_sample_data import SampleDataGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_document_processor():
    """Test the document processor functionality."""
    logger.info("Testing Document Processor...")
    
    try:
        # Initialize document processor
        processor = DocumentProcessor()
        
        # Check available processors
        available = processor.get_available_processors()
        logger.info(f"Available processors: {available}")
        
        return True
    except Exception as e:
        logger.error(f"Document processor test failed: {e}")
        return False


def test_pdf_processing():
    """Test PDF processing functionality."""
    logger.info("Testing PDF Processing...")
    
    try:
        # Generate sample PDF data
        generator = SampleDataGenerator("test_data")
        generator.generate_pdf_data(num_files=2, num_nodes=1)
        
        # Test PDF processing
        pdf_processor = PDFProcessor()
        
        # Find generated PDF files
        pdf_files = list(Path("test_data/node1/pdfs").glob("*.pdf"))
        
        if not pdf_files:
            logger.warning("No PDF files found for testing")
            return False
        
        # Process first PDF
        pdf_path = pdf_files[0]
        logger.info(f"Processing PDF: {pdf_path}")
        
        content = pdf_processor.process_pdf(str(pdf_path))
        
        logger.info(f"Extracted text length: {len(content.text)}")
        logger.info(f"Number of tables: {len(content.tables)}")
        logger.info(f"Number of images: {len(content.images)}")
        logger.info(f"Metadata keys: {list(content.metadata.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"PDF processing test failed: {e}")
        return False


def test_web_scraping():
    """Test web scraping functionality."""
    logger.info("Testing Web Scraping...")
    
    try:
        # Generate sample web data
        generator = SampleDataGenerator("test_data")
        generator.generate_web_data(num_files=2, num_nodes=1)
        
        # Test web scraping
        scraper = WebScraper()
        
        # Find generated HTML files
        html_files = list(Path("test_data/node1/web").glob("*.html"))
        
        if not html_files:
            logger.warning("No HTML files found for testing")
            return False
        
        # Process first HTML file
        html_path = html_files[0]
        logger.info(f"Processing HTML: {html_path}")
        
        content = scraper.scrape_url(f"file://{html_path.absolute()}", method='beautifulsoup')
        
        logger.info(f"Title: {content.title}")
        logger.info(f"Text length: {len(content.text)}")
        logger.info(f"Number of links: {len(content.links)}")
        logger.info(f"Number of images: {len(content.images)}")
        logger.info(f"Number of tables: {len(content.tables)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Web scraping test failed: {e}")
        return False


def test_data_loader_integration():
    """Test data loader integration with document processing."""
    logger.info("Testing Data Loader Integration...")
    
    try:
        # Test PDF data loading
        pdf_dataset = UnstructuredDataset("test_data/node1/pdfs", "pdf")
        logger.info(f"PDF dataset size: {len(pdf_dataset)}")
        
        # Test web data loading
        web_dataset = UnstructuredDataset("test_data/node1/web", "web")
        logger.info(f"Web dataset size: {len(web_dataset)}")
        
        # Test document data loading
        document_dataset = UnstructuredDataset("test_data/node1", "document")
        logger.info(f"Document dataset size: {len(document_dataset)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Data loader integration test failed: {e}")
        return False


def test_ocr_processing():
    """Test OCR processing functionality."""
    logger.info("Testing OCR Processing...")
    
    try:
        # Generate sample image data
        generator = SampleDataGenerator("test_data")
        generator.generate_image_data(num_files=2, num_nodes=1)
        
        # Test OCR processing
        ocr_processor = OCRProcessor()
        
        # Find generated image files
        image_files = list(Path("test_data/node1/images").glob("*.png"))
        
        if not image_files:
            logger.warning("No image files found for testing")
            return False
        
        # Process first image
        image_path = image_files[0]
        logger.info(f"Processing image: {image_path}")
        
        text, confidence = ocr_processor.extract_text_with_confidence(str(image_path))
        
        logger.info(f"Extracted text: {text[:100]}...")
        logger.info(f"Confidence: {confidence:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"OCR processing test failed: {e}")
        return False


def cleanup_test_data():
    """Clean up test data."""
    import shutil
    
    test_data_path = Path("test_data")
    if test_data_path.exists():
        shutil.rmtree(test_data_path)
        logger.info("Cleaned up test data")


def main():
    """Main test function."""
    logger.info("Starting Document Processing Tests")
    
    tests = [
        ("Document Processor", test_document_processor),
        ("PDF Processing", test_pdf_processing),
        ("Web Scraping", test_web_scraping),
        ("OCR Processing", test_ocr_processing),
        ("Data Loader Integration", test_data_loader_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} Test")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results[test_name] = success
            status = "PASSED" if success else "FAILED"
            logger.info(f"{test_name} Test: {status}")
        except Exception as e:
            logger.error(f"{test_name} Test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "PASSED" if success else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    # Cleanup
    cleanup_test_data()
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 