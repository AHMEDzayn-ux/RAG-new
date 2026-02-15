"""
Download sample PDFs for testing

Downloads public domain and sample PDF documents for testing the RAG system.
"""

import urllib.request
from pathlib import Path

def download_pdfs():
    """Download sample PDF files"""
    
    documents_dir = Path("../documents")
    documents_dir.mkdir(exist_ok=True)
    
    pdfs = [
        {
            "name": "university_guide.pdf",
            "url": "https://www.adobe.com/support/products/enterprise/knowledgecenter/media/c4611_sample_explain.pdf",
            "description": "University admissions guide sample"
        },
        {
            "name": "sample_faq.pdf",
            "url": "https://www.africau.edu/images/default/sample.pdf",
            "description": "Sample FAQ document"
        }
    ]
    
    print("Downloading sample PDFs...")
    print("=" * 60)
    
    for pdf in pdfs:
        try:
            output_path = documents_dir / pdf["name"]
            print(f"\nDownloading: {pdf['description']}")
            print(f"URL: {pdf['url']}")
            print(f"Saving to: {output_path}")
            
            urllib.request.urlretrieve(pdf["url"], str(output_path))
            
            # Check file size
            size = output_path.stat().st_size
            print(f"✓ Downloaded successfully ({size:,} bytes)")
            
        except Exception as e:
            print(f"✗ Failed to download {pdf['name']}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"\nPDFs saved in: {documents_dir.absolute()}")

if __name__ == "__main__":
    download_pdfs()
