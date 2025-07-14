import xml.etree.ElementTree as ET
import json
import logging
from typing import List, Dict, Any, Optional
import re

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def extract_tag_text(tag_elem, ns):
    value_elem = tag_elem.find('rdf:value', ns)
    if value_elem is not None and value_elem.text:
        return clean_text(value_elem.text)
    value_elem = tag_elem.find(f"{{{ns['rdf']}}}value")
    if value_elem is not None and value_elem.text:
        return clean_text(value_elem.text)
    return clean_text(tag_elem.text)

def parse_zotero_rdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse Zotero RDF/XML export file into structured paper metadata.
    
    Args:
        file_path: Path to the Zotero RDF export file
        
    Returns:
        List of paper dictionaries with metadata
    """
    ns = {
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'dc': 'http://purl.org/dc/elements/1.1/',
        'dcterms': 'http://purl.org/dc/terms/',
        'z': 'http://www.zotero.org/namespaces/export#',
        'bibo': 'http://purl.org/net/biblio#',
        'foaf': 'http://xmlns.com/foaf/0.1/',
    }
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        logger.error(f"Failed to parse XML file: {e}")
        return []
    
    items = []
    # Find all paper elements
    paper_elements = []
    for elem in root:
        if elem.tag.endswith('Article') or elem.tag.endswith('Document') or elem.tag.endswith('Description'):
            paper_elements.append(elem)
    for elem in paper_elements:
        try:
            # Extract basic metadata
            title_elem = elem.find('dc:title', ns)
            title = clean_text(title_elem.text) if title_elem is not None and title_elem.text else ""
            
            # Try different abstract formats
            abstract_elem = elem.find('z:abstractNote', ns)
            if abstract_elem is None or not abstract_elem.text:
                abstract_elem = elem.find('dcterms:abstract', ns)
            abstract = clean_text(abstract_elem.text) if abstract_elem is not None and abstract_elem.text else ""
            
            # Extract tags from direct children
            tags = [clean_text(tag.text) for tag in elem.findall('z:tag', ns) if tag.text]
            for tag_elem in elem.findall('z:AutomaticTag', ns):
                tag_val = extract_tag_text(tag_elem, ns)
                if tag_val:
                    tags.append(tag_val)
            # Extract tags from <dc:subject> children
            for subj in elem.findall('dc:subject', ns):
                for tag_elem in subj.findall('z:tag', ns):
                    tag_val = clean_text(tag_elem.text)
                    if tag_val:
                        tags.append(tag_val)
                for tag_elem in subj.findall('z:AutomaticTag', ns):
                    tag_val = extract_tag_text(tag_elem, ns)
                    if tag_val:
                        tags.append(tag_val)
            
            # Extract notes
            notes_elem = elem.find('z:note', ns)
            notes = clean_text(notes_elem.text) if notes_elem is not None and notes_elem.text else ""
            
            # Extract collections
            collections = [clean_text(col.text) for col in elem.findall('z:collection', ns) if col.text]
            
            # Extract related items
            related = [clean_text(rel_elem.text) for rel_elem in elem.findall('z:relatedItem', ns) if rel_elem.text]
            
            # Extract additional metadata
            authors = [clean_text(author_elem.text) for author_elem in elem.findall('dc:creator', ns) if author_elem.text]
            
            # Extract publication year
            date_elem = elem.find('dc:date', ns)
            year = None
            if date_elem is not None and date_elem.text:
                try:
                    year = int(date_elem.text[:4])
                except ValueError:
                    pass
            
            # Extract DOI
            doi_elem = elem.find('bibo:doi', ns)
            doi = clean_text(doi_elem.text) if doi_elem is not None and doi_elem.text else ""
            
            # Extract URL
            url_elem = elem.find('z:url', ns)
            url = clean_text(url_elem.text) if url_elem is not None and url_elem.text else ""
            
            # Only include items with sufficient metadata
            if title and abstract and len(abstract) > 50:
                item = {
                    'title': title,
                    'abstract': abstract,
                    'tags': tags,
                    'notes': notes,
                    'collections': collections,
                    'related': related,
                    'authors': authors,
                    'year': year,
                    'doi': doi,
                    'url': url,
                    'item_type': 'journalArticle',  # Default type
                }
                items.append(item)
            else:
                logger.warning(f"Skipping item with insufficient metadata: {title[:50]}...")
                
        except Exception as e:
            logger.error(f"Error parsing item: {e}")
            continue
    
    logger.info(f"Successfully parsed {len(items)} papers from {file_path}")
    return items

def parse_zotero_json(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse Zotero JSON export file (alternative format).
    
    Args:
        file_path: Path to the Zotero JSON export file
        
    Returns:
        List of paper dictionaries with metadata
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"Failed to parse JSON file: {e}")
        return []
    
    items = []
    for item in data:
        try:
            # Extract metadata from JSON structure
            title = clean_text(item.get('title', ''))
            abstract = clean_text(item.get('abstractNote', ''))
            
            # Extract tags
            tags = []
            for tag in item.get('tags', []):
                if isinstance(tag, dict):
                    tag_name = tag.get('tag', '')
                else:
                    tag_name = str(tag)
                if tag_name:
                    tags.append(clean_text(tag_name))
            
            # Extract notes
            notes = clean_text(item.get('note', ''))
            
            # Extract collections
            collections = []
            for collection in item.get('collections', []):
                if isinstance(collection, dict):
                    col_name = collection.get('name', '')
                else:
                    col_name = str(collection)
                if col_name:
                    collections.append(clean_text(col_name))
            
            # Extract related items
            related = []
            for rel_item in item.get('relatedItems', []):
                if isinstance(rel_item, dict):
                    rel_title = rel_item.get('title', '')
                else:
                    rel_title = str(rel_item)
                if rel_title:
                    related.append(clean_text(rel_title))
            
            # Extract authors
            authors = []
            for creator in item.get('creators', []):
                if isinstance(creator, dict):
                    author_name = creator.get('name', '')
                else:
                    author_name = str(creator)
                if author_name:
                    authors.append(clean_text(author_name))
            
            # Extract year
            year = None
            date = item.get('date', '')
            if date:
                try:
                    year = int(date[:4])
                except ValueError:
                    pass
            
            # Extract DOI and URL
            doi = clean_text(item.get('DOI', ''))
            url = clean_text(item.get('url', ''))
            
            # Only include items with sufficient metadata
            if title and abstract and len(abstract) > 50:
                item_dict = {
                    'title': title,
                    'abstract': abstract,
                    'tags': tags,
                    'notes': notes,
                    'collections': collections,
                    'related': related,
                    'authors': authors,
                    'year': year,
                    'doi': doi,
                    'url': url,
                    'item_type': item.get('itemType', 'journalArticle'),
                }
                items.append(item_dict)
            else:
                logger.warning(f"Skipping item with insufficient metadata: {title[:50]}...")
                
        except Exception as e:
            logger.error(f"Error parsing JSON item: {e}")
            continue
    
    logger.info(f"Successfully parsed {len(items)} papers from {file_path}")
    return items

def parse_zotero_export(file_path: str) -> List[Dict[str, Any]]:
    """
    Parse Zotero export file (supports both RDF and JSON formats).
    
    Args:
        file_path: Path to the Zotero export file
        
    Returns:
        List of paper dictionaries with metadata
    """
    if file_path.endswith('.rdf') or file_path.endswith('.xml'):
        return parse_zotero_rdf(file_path)
    elif file_path.endswith('.json'):
        return parse_zotero_json(file_path)
    else:
        logger.error(f"Unsupported file format: {file_path}")
        return []

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = 'data/example_zotero_export.rdf'
    
    papers = parse_zotero_export(file_path)
    print(json.dumps(papers[:2], indent=2))  # Show first 2 papers as example
