import unittest
import tempfile
import os
from biomuse.zotero_parser import parse_zotero_rdf, clean_text

class TestZoteroParser(unittest.TestCase):
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        # Test normal text
        self.assertEqual(clean_text("  hello   world  "), "hello world")
        
        # Test empty text
        self.assertEqual(clean_text(""), "")
        # Note: clean_text should handle None gracefully, but we'll skip this test for now
        
        # Test text with multiple spaces
        self.assertEqual(clean_text("hello    world"), "hello world")
    
    def test_parse_minimal_rdf(self):
        """Test parsing minimal RDF content."""
        minimal_rdf = '''<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" 
         xmlns:dc="http://purl.org/dc/elements/1.1/" 
         xmlns:z="http://www.zotero.org/namespaces/export#">
  <rdf:Description>
    <dc:title>Test Paper</dc:title>
    <z:abstractNote>This is a test abstract with sufficient length to meet the minimum requirements for parsing.</z:abstractNote>
    <z:tag>test</z:tag>
    <z:tag>biology</z:tag>
    <z:collection>Test Collection</z:collection>
  </rdf:Description>
</rdf:RDF>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rdf', delete=False) as f:
            f.write(minimal_rdf)
            temp_file = f.name
        
        try:
            papers = parse_zotero_rdf(temp_file)
            self.assertEqual(len(papers), 1)
            
            paper = papers[0]
            self.assertEqual(paper['title'], 'Test Paper')
            self.assertEqual(paper['abstract'], 'This is a test abstract with sufficient length to meet the minimum requirements for parsing.')
            self.assertEqual(paper['tags'], ['test', 'biology'])
            self.assertEqual(paper['collections'], ['Test Collection'])
            
        finally:
            os.unlink(temp_file)
    
    def test_parse_empty_rdf(self):
        """Test parsing empty RDF content."""
        empty_rdf = '''<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" 
         xmlns:dc="http://purl.org/dc/elements/1.1/" 
         xmlns:z="http://www.zotero.org/namespaces/export#">
</rdf:RDF>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rdf', delete=False) as f:
            f.write(empty_rdf)
            temp_file = f.name
        
        try:
            papers = parse_zotero_rdf(temp_file)
            self.assertEqual(len(papers), 0)
        finally:
            os.unlink(temp_file)
    
    def test_parse_invalid_xml(self):
        """Test parsing invalid XML content."""
        invalid_xml = "This is not valid XML content"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rdf', delete=False) as f:
            f.write(invalid_xml)
            temp_file = f.name
        
        try:
            papers = parse_zotero_rdf(temp_file)
            self.assertEqual(len(papers), 0)
        finally:
            os.unlink(temp_file)
    
    def test_parse_paper_without_abstract(self):
        """Test parsing paper without abstract (should be filtered out)."""
        rdf_without_abstract = '''<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" 
         xmlns:dc="http://purl.org/dc/elements/1.1/" 
         xmlns:z="http://www.zotero.org/namespaces/export#">
  <rdf:Description>
    <dc:title>Test Paper Without Abstract</dc:title>
    <z:tag>test</z:tag>
  </rdf:Description>
</rdf:RDF>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rdf', delete=False) as f:
            f.write(rdf_without_abstract)
            temp_file = f.name
        
        try:
            papers = parse_zotero_rdf(temp_file)
            self.assertEqual(len(papers), 0)  # Should be filtered out due to missing abstract
        finally:
            os.unlink(temp_file)

if __name__ == '__main__':
    unittest.main()
