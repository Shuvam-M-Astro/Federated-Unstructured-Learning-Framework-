"""
Document processing module for PDFs and web scraping in federated learning.

This module provides comprehensive document processing capabilities including:
- PDF text, image, and table extraction
- Web scraping with multiple methods
- OCR processing for scanned documents
- Document metadata extraction
- Content cleaning and normalization
"""

import os
import re
import json
import logging
import time
import hashlib
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from PIL import Image
import requests
from bs4 import BeautifulSoup
import lxml.etree as etree

# PDF Processing
try:
    import PyPDF2
    import pdfplumber
    from pdf2image import convert_from_path
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    warnings.warn("PDF processing libraries not available. Install PyPDF2, pdfplumber, pdf2image")

# OCR Processing
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    warnings.warn("OCR libraries not available. Install pytesseract")

# Web Scraping
try:
    import trafilatura
    from newspaper import Article
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False
    warnings.warn("Web scraping libraries not available. Install trafilatura, newspaper3k")

from utils.constants import (
    PDF_EXTRACTION_METHODS, PDF_MAX_PAGES, PDF_TEXT_ENCODING, PDF_IMAGE_DPI,
    WEB_SCRAPING_METHODS, WEB_SCRAPING_TIMEOUT, WEB_SCRAPING_RETRY_ATTEMPTS,
    WEB_SCRAPING_DELAY, WEB_SCRAPING_USER_AGENT, WEB_SCRAPING_EXCLUDE_PATTERNS,
    OCR_LANGUAGES, OCR_CONFIDENCE_THRESHOLD, DOCUMENT_MAX_SIZE_MB
)

logger = logging.getLogger(__name__)


@dataclass
class DocumentContent:
    """Container for extracted document content."""
    text: str
    tables: List[pd.DataFrame]
    images: List[Image.Image]
    metadata: Dict[str, Any]
    raw_html: Optional[str] = None
    cleaned_html: Optional[str] = None
    extraction_method: str = "unknown"
    confidence: float = 1.0


@dataclass
class WebPageContent:
    """Container for scraped web page content."""
    title: str
    text: str
    html: str
    metadata: Dict[str, Any]
    links: List[str]
    images: List[str]
    tables: List[pd.DataFrame]
    extraction_method: str = "unknown"
    url: str = ""


class PDFProcessor:
    """Process PDF documents for federated learning."""
    
    def __init__(self, 
                 extraction_methods: List[str] = None,
                 max_pages: int = PDF_MAX_PAGES,
                 image_dpi: int = PDF_IMAGE_DPI,
                 table_detection_method: str = 'stream'):
        """Initialize PDF processor.
        
        Args:
            extraction_methods: Methods to use for extraction
            max_pages: Maximum pages to process
            image_dpi: DPI for image extraction
            table_detection_method: Method for table detection
        """
        if not PDF_AVAILABLE:
            raise ImportError("PDF processing libraries not available")
        
        self.extraction_methods = extraction_methods or ['text', 'tables', 'metadata']
        self.max_pages = max_pages
        self.image_dpi = image_dpi
        self.table_detection_method = table_detection_method
        
        # Validate extraction methods
        for method in self.extraction_methods:
            if method not in PDF_EXTRACTION_METHODS:
                raise ValueError(f"Unsupported extraction method: {method}")
    
    def process_pdf(self, pdf_path: str) -> DocumentContent:
        """Process a PDF file and extract content.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            DocumentContent object with extracted information
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if pdf_path.stat().st_size > DOCUMENT_MAX_SIZE_MB * 1024 * 1024:
            raise ValueError(f"PDF file too large: {pdf_path.stat().st_size / 1024 / 1024:.1f}MB")
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        content = DocumentContent(
            text="",
            tables=[],
            images=[],
            metadata={},
            extraction_method="pdf"
        )
        
        try:
            # Extract metadata
            if 'metadata' in self.extraction_methods:
                content.metadata = self._extract_metadata(pdf_path)
            
            # Extract text
            if 'text' in self.extraction_methods:
                content.text = self._extract_text(pdf_path)
            
            # Extract tables
            if 'tables' in self.extraction_methods:
                content.tables = self._extract_tables(pdf_path)
            
            # Extract images
            if 'images' in self.extraction_methods:
                content.images = self._extract_images(pdf_path)
            
            logger.info(f"Successfully processed PDF: {pdf_path}")
            return content
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def _extract_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract PDF metadata."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                info = pdf_reader.metadata
                
                metadata = {
                    'num_pages': len(pdf_reader.pages),
                    'file_size': pdf_path.stat().st_size,
                    'file_path': str(pdf_path),
                    'extraction_timestamp': time.time()
                }
                
                if info:
                    metadata.update({
                        'title': info.get('/Title', ''),
                        'author': info.get('/Author', ''),
                        'subject': info.get('/Subject', ''),
                        'creator': info.get('/Creator', ''),
                        'producer': info.get('/Producer', ''),
                        'creation_date': info.get('/CreationDate', ''),
                        'modification_date': info.get('/ModDate', '')
                    })
                
                return metadata
        except Exception as e:
            logger.warning(f"Error extracting metadata from {pdf_path}: {e}")
            return {'file_path': str(pdf_path), 'error': str(e)}
    
    def _extract_text(self, pdf_path: Path) -> str:
        """Extract text from PDF."""
        try:
            text_content = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages[:self.max_pages]):
                    text = page.extract_text()
                    if text:
                        text_content.append(f"--- Page {page_num + 1} ---\n{text}\n")
            
            return "\n".join(text_content)
        except Exception as e:
            logger.warning(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def _extract_tables(self, pdf_path: Path) -> List[pd.DataFrame]:
        """Extract tables from PDF."""
        try:
            tables = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages[:self.max_pages]):
                    page_tables = page.extract_tables()
                    for table_num, table in enumerate(page_tables):
                        if table:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            df['_page'] = page_num + 1
                            df['_table'] = table_num + 1
                            tables.append(df)
            
            return tables
        except Exception as e:
            logger.warning(f"Error extracting tables from {pdf_path}: {e}")
            return []
    
    def _extract_images(self, pdf_path: Path) -> List[Image.Image]:
        """Extract images from PDF."""
        try:
            images = convert_from_path(pdf_path, dpi=self.image_dpi)
            return images
        except Exception as e:
            logger.warning(f"Error extracting images from {pdf_path}: {e}")
            return []


class WebScraper:
    """Scrape web pages for federated learning."""
    
    def __init__(self,
                 timeout: int = WEB_SCRAPING_TIMEOUT,
                 retry_attempts: int = WEB_SCRAPING_RETRY_ATTEMPTS,
                 delay: float = WEB_SCRAPING_DELAY,
                 user_agent: str = WEB_SCRAPING_USER_AGENT,
                 max_pages: int = 100):
        """Initialize web scraper.
        
        Args:
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
            delay: Delay between requests
            user_agent: User agent string
            max_pages: Maximum pages to scrape
        """
        if not WEB_SCRAPING_AVAILABLE:
            raise ImportError("Web scraping libraries not available")
        
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.delay = delay
        self.user_agent = user_agent
        self.max_pages = max_pages
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': user_agent})
    
    def scrape_url(self, url: str, method: str = 'trafilatura') -> WebPageContent:
        """Scrape a single URL.
        
        Args:
            url: URL to scrape
            method: Scraping method to use
            
        Returns:
            WebPageContent object with scraped information
        """
        logger.info(f"Scraping URL: {url}")
        
        try:
            if method == 'trafilatura':
                return self._scrape_with_trafilatura(url)
            elif method == 'newspaper':
                return self._scrape_with_newspaper(url)
            elif method == 'beautifulsoup':
                return self._scrape_with_beautifulsoup(url)
            else:
                raise ValueError(f"Unsupported scraping method: {method}")
                
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {e}")
            raise
    
    def scrape_multiple_urls(self, urls: List[str], method: str = 'trafilatura') -> List[WebPageContent]:
        """Scrape multiple URLs.
        
        Args:
            urls: List of URLs to scrape
            method: Scraping method to use
            
        Returns:
            List of WebPageContent objects
        """
        results = []
        
        for i, url in enumerate(urls[:self.max_pages]):
            try:
                content = self.scrape_url(url, method)
                results.append(content)
                
                # Add delay between requests
                if i < len(urls) - 1:
                    time.sleep(self.delay)
                    
            except Exception as e:
                logger.warning(f"Failed to scrape {url}: {e}")
                continue
        
        return results
    
    def _scrape_with_trafilatura(self, url: str) -> WebPageContent:
        """Scrape using trafilatura library."""
        try:
            # Download and extract content
            downloaded = trafilatura.fetch_url(url)
            if downloaded is None:
                raise ValueError(f"Failed to download content from {url}")
            
            # Extract main content
            text = trafilatura.extract(downloaded, include_formatting=True)
            if text is None:
                text = ""
            
            # Extract metadata
            metadata = trafilatura.extract_metadata(downloaded)
            if metadata is None:
                metadata = {}
            
            # Extract links
            links = trafilatura.extract_links(downloaded)
            if links is None:
                links = []
            
            # Parse HTML for additional information
            soup = BeautifulSoup(downloaded, 'lxml')
            
            # Extract title
            title = metadata.get('title', '')
            if not title:
                title_tag = soup.find('title')
                title = title_tag.get_text() if title_tag else ''
            
            # Extract tables
            tables = []
            for table in soup.find_all('table'):
                try:
                    df = pd.read_html(str(table))[0]
                    tables.append(df)
                except:
                    continue
            
            # Extract image URLs
            images = []
            for img in soup.find_all('img'):
                src = img.get('src')
                if src:
                    images.append(urljoin(url, src))
            
            return WebPageContent(
                title=title,
                text=text,
                html=downloaded,
                metadata=metadata,
                links=links,
                images=images,
                tables=tables,
                extraction_method="trafilatura",
                url=url
            )
            
        except Exception as e:
            logger.error(f"Error with trafilatura scraping {url}: {e}")
            raise
    
    def _scrape_with_newspaper(self, url: str) -> WebPageContent:
        """Scrape using newspaper3k library."""
        try:
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            
            # Extract tables from HTML
            tables = []
            if hasattr(article, 'html') and article.html:
                soup = BeautifulSoup(article.html, 'lxml')
                for table in soup.find_all('table'):
                    try:
                        df = pd.read_html(str(table))[0]
                        tables.append(df)
                    except:
                        continue
            
            # Extract image URLs
            images = []
            if hasattr(article, 'images') and article.images:
                images = article.images
            
            return WebPageContent(
                title=article.title or '',
                text=article.text or '',
                html=article.html or '',
                metadata={
                    'authors': article.authors,
                    'publish_date': str(article.publish_date) if article.publish_date else '',
                    'summary': article.summary,
                    'keywords': article.keywords,
                    'meta_keywords': article.meta_keywords,
                    'meta_description': article.meta_description,
                    'meta_lang': article.meta_lang,
                    'meta_favicon': article.meta_favicon,
                    'meta_img': article.meta_img,
                    'canonical_link': article.canonical_link,
                    'tags': article.tags,
                    'movies': article.movies
                },
                links=[],
                images=images,
                tables=tables,
                extraction_method="newspaper",
                url=url
            )
            
        except Exception as e:
            logger.error(f"Error with newspaper scraping {url}: {e}")
            raise
    
    def _scrape_with_beautifulsoup(self, url: str) -> WebPageContent:
        """Scrape using BeautifulSoup library."""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text() if title_tag else ''
            
            # Extract text (remove script and style elements)
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            
            # Clean text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Extract links
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('http'):
                    links.append(href)
                elif href.startswith('/'):
                    links.append(urljoin(url, href))
            
            # Extract tables
            tables = []
            for table in soup.find_all('table'):
                try:
                    df = pd.read_html(str(table))[0]
                    tables.append(df)
                except:
                    continue
            
            # Extract image URLs
            images = []
            for img in soup.find_all('img'):
                src = img.get('src')
                if src:
                    images.append(urljoin(url, src))
            
            return WebPageContent(
                title=title,
                text=text,
                html=str(soup),
                metadata={
                    'url': url,
                    'status_code': response.status_code,
                    'content_type': response.headers.get('content-type', ''),
                    'content_length': len(response.content)
                },
                links=links,
                images=images,
                tables=tables,
                extraction_method="beautifulsoup",
                url=url
            )
            
        except Exception as e:
            logger.error(f"Error with BeautifulSoup scraping {url}: {e}")
            raise


class OCRProcessor:
    """Process OCR for scanned documents and images."""
    
    def __init__(self, 
                 languages: List[str] = None,
                 confidence_threshold: float = OCR_CONFIDENCE_THRESHOLD):
        """Initialize OCR processor.
        
        Args:
            languages: Languages for OCR
            confidence_threshold: Minimum confidence threshold
        """
        if not OCR_AVAILABLE:
            raise ImportError("OCR libraries not available")
        
        self.languages = languages or ['en']
        self.confidence_threshold = confidence_threshold
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text
        """
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(
                image, 
                lang='+'.join(self.languages),
                config='--psm 6'
            )
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from image {image_path}: {e}")
            return ""
    
    def extract_text_with_confidence(self, image_path: str) -> Tuple[str, float]:
        """Extract text with confidence score.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (text, confidence_score)
        """
        try:
            image = Image.open(image_path)
            data = pytesseract.image_to_data(
                image, 
                lang='+'.join(self.languages),
                config='--psm 6',
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and confidence
            text_parts = []
            total_confidence = 0
            valid_words = 0
            
            for i, conf in enumerate(data['conf']):
                if conf > self.confidence_threshold * 100:
                    text_parts.append(data['text'][i])
                    total_confidence += conf
                    valid_words += 1
            
            text = ' '.join(text_parts)
            avg_confidence = total_confidence / valid_words if valid_words > 0 else 0
            
            return text, avg_confidence / 100.0
            
        except Exception as e:
            logger.error(f"Error extracting text with confidence from {image_path}: {e}")
            return "", 0.0


class DocumentProcessor:
    """Main document processor for federated learning."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize document processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize processors
        self.pdf_processor = None
        self.web_scraper = None
        self.ocr_processor = None
        
        if PDF_AVAILABLE:
            self.pdf_processor = PDFProcessor(
                extraction_methods=self.config.get('pdf_extraction_methods', ['text', 'tables', 'metadata']),
                max_pages=self.config.get('pdf_max_pages', PDF_MAX_PAGES),
                image_dpi=self.config.get('pdf_image_dpi', PDF_IMAGE_DPI)
            )
        
        if WEB_SCRAPING_AVAILABLE:
            self.web_scraper = WebScraper(
                timeout=self.config.get('web_timeout', WEB_SCRAPING_TIMEOUT),
                retry_attempts=self.config.get('web_retry_attempts', WEB_SCRAPING_RETRY_ATTEMPTS),
                delay=self.config.get('web_delay', WEB_SCRAPING_DELAY)
            )
        
        if OCR_AVAILABLE:
            self.ocr_processor = OCRProcessor(
                languages=self.config.get('ocr_languages', ['en']),
                confidence_threshold=self.config.get('ocr_confidence_threshold', OCR_CONFIDENCE_THRESHOLD)
            )
    
    def process_document(self, file_path: str) -> DocumentContent:
        """Process a document file.
        
        Args:
            file_path: Path to document file
            
        Returns:
            DocumentContent object
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.pdf':
            if not self.pdf_processor:
                raise ImportError("PDF processing not available")
            return self.pdf_processor.process_pdf(file_path)
        
        elif file_extension in ['.html', '.htm']:
            # Treat as local HTML file
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Extract text
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            
            # Clean text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Extract tables
            tables = []
            for table in soup.find_all('table'):
                try:
                    df = pd.read_html(str(table))[0]
                    tables.append(df)
                except:
                    continue
            
            return DocumentContent(
                text=text,
                tables=tables,
                images=[],
                metadata={'file_path': str(file_path), 'file_type': 'html'},
                raw_html=html_content,
                cleaned_html=str(soup),
                extraction_method="html_parser"
            )
        
        else:
            # Try OCR for image files
            if self.ocr_processor and file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                text, confidence = self.ocr_processor.extract_text_with_confidence(file_path)
                return DocumentContent(
                    text=text,
                    tables=[],
                    images=[Image.open(file_path)],
                    metadata={
                        'file_path': str(file_path),
                        'file_type': 'image',
                        'ocr_confidence': confidence
                    },
                    extraction_method="ocr",
                    confidence=confidence
                )
            
            # Fallback to text processing
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                return DocumentContent(
                    text=text,
                    tables=[],
                    images=[],
                    metadata={'file_path': str(file_path), 'file_type': 'text'},
                    extraction_method="text_reader"
                )
            except Exception as e:
                logger.error(f"Error processing document {file_path}: {e}")
                raise
    
    def scrape_website(self, url: str, method: str = 'trafilatura') -> WebPageContent:
        """Scrape a website.
        
        Args:
            url: URL to scrape
            method: Scraping method
            
        Returns:
            WebPageContent object
        """
        if not self.web_scraper:
            raise ImportError("Web scraping not available")
        
        return self.web_scraper.scrape_url(url, method)
    
    def get_available_processors(self) -> Dict[str, bool]:
        """Get information about available processors.
        
        Returns:
            Dictionary of processor availability
        """
        return {
            'pdf_processing': PDF_AVAILABLE,
            'web_scraping': WEB_SCRAPING_AVAILABLE,
            'ocr_processing': OCR_AVAILABLE
        } 