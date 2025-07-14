import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

# Domain mapping from task_generator.py
DOMAIN_MAPPING = {
    'cancer_evolution': ['cancer', 'evolution', 'genomics'],
    'protein_structure': ['protein', 'structure', 'prediction'],
    'gene_regulation': ['gene', 'regulation', 'transcription']
}

def clean_text(text):
    return text.strip() if text else ''

def extract_tag_text(tag_elem, ns):
    # Handles <z:AutomaticTag><rdf:value>TAG</rdf:value></z:AutomaticTag> with namespace
    value_elem = tag_elem.find('rdf:value', ns)
    if value_elem is not None and value_elem.text:
        return clean_text(value_elem.text)
    # Try with full namespace if above fails
    value_elem = tag_elem.find(f"{{{ns['rdf']}}}value")
    if value_elem is not None and value_elem.text:
        return clean_text(value_elem.text)
    return clean_text(tag_elem.text)

def analyze_rdf(rdf_path):
    ns = {
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'dc': 'http://purl.org/dc/elements/1.1/',
        'dcterms': 'http://purl.org/dc/terms/',
        'z': 'http://www.zotero.org/namespaces/export#',
        'bibo': 'http://purl.org/net/biblio#',
        'foaf': 'http://xmlns.com/foaf/0.1/',
    }
    tree = ET.parse(rdf_path)
    root = tree.getroot()
    papers = []
    tag_counter = Counter()
    tag_papers = defaultdict(list)
    collection_counter = Counter()
    collection_papers = defaultdict(list)
    domain_match_counter = Counter()
    abstracts = 0
    notes = 0
    # Index all paper elements by rdf:about
    paper_elements = {}
    for elem in root:
        if elem.tag.endswith('Article') or elem.tag.endswith('Document') or elem.tag.endswith('Description'):
            about = elem.attrib.get(f'{{{ns["rdf"]}}}about')
            if about:
                paper_elements[about] = elem
    # Index tags from <rdf:Description> by rdf:about
    tag_map = defaultdict(list)
    for desc in root.findall('rdf:Description', ns):
        about = desc.attrib.get(f'{{{ns["rdf"]}}}about')
        if not about:
            continue
        # <z:tag>
        for tag_elem in desc.findall('z:tag', ns):
            tag_val = clean_text(tag_elem.text)
            if tag_val:
                tag_map[about].append(tag_val)
        # <z:AutomaticTag><rdf:value>...</rdf:value></z:AutomaticTag>
        for tag_elem in desc.findall('z:AutomaticTag', ns):
            tag_val = extract_tag_text(tag_elem, ns)
            if tag_val:
                tag_map[about].append(tag_val)
    # Now process each paper
    for about, elem in paper_elements.items():
        title = clean_text(elem.findtext('dc:title', default='', namespaces=ns))
        abstract = clean_text(elem.findtext('z:abstractNote', default='', namespaces=ns))
        if not abstract:
            abstract = clean_text(elem.findtext('dcterms:abstract', default='', namespaces=ns))
        note = clean_text(elem.findtext('z:note', default='', namespaces=ns))
        # Tags from paper element
        tags = [clean_text(tag.text) for tag in elem.findall('z:tag', ns) if tag.text]
        for tag_elem in elem.findall('z:AutomaticTag', ns):
            tag_val = extract_tag_text(tag_elem, ns)
            if tag_val:
                tags.append(tag_val)
        # Tags from <dc:subject> children
        for subj in elem.findall('dc:subject', ns):
            for tag_elem in subj.findall('z:tag', ns):
                tag_val = clean_text(tag_elem.text)
                if tag_val:
                    tags.append(tag_val)
            for tag_elem in subj.findall('z:AutomaticTag', ns):
                tag_val = extract_tag_text(tag_elem, ns)
                if tag_val:
                    tags.append(tag_val)
        # Tags from <rdf:Description>
        tags += tag_map.get(about, [])
        # Collections
        collections = [clean_text(col.text) for col in elem.findall('z:collection', ns) if col.text]
        # Count
        if tags:
            for tag in tags:
                tag_counter[tag] += 1
                tag_papers[tag].append(title)
        if collections:
            for col in collections:
                collection_counter[col] += 1
                collection_papers[col].append(title)
        if abstract:
            abstracts += 1
        if note:
            notes += 1
        # Domain mapping
        all_text = ' '.join(collections + tags).lower()
        for domain, keywords in DOMAIN_MAPPING.items():
            if any(keyword in all_text for keyword in keywords):
                domain_match_counter[domain] += 1
        papers.append({
            'title': title,
            'tags': tags,
            'collections': collections,
            'abstract': abstract,
            'note': note
        })
    # Report
    print(f"Total papers: {len(papers)}")
    print(f"Papers with tags: {sum(1 for p in papers if p['tags'])}")
    print(f"Unique tags: {len(tag_counter)}")
    print(f"Tags shared by >1 paper: {[tag for tag, count in tag_counter.items() if count > 1]}")
    print(f"Papers with abstracts: {abstracts}")
    print(f"Papers with notes: {notes}")
    print(f"Papers with collections: {sum(1 for p in papers if p['collections'])}")
    print(f"Collections shared by >1 paper: {[col for col, count in collection_counter.items() if count > 1]}")
    print(f"Domain matches: {dict(domain_match_counter)}")
    # Warnings
    if not any(count > 1 for count in tag_counter.values()):
        print("WARNING: No tags are shared by more than one paper. Retrieval tasks will not be generated.")
    if notes == 0:
        print("WARNING: No papers have notes. Summarization tasks will not be generated.")
    if not domain_match_counter:
        print("WARNING: No papers match the domain mapping for classification tasks.")
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_rdf_metadata.py <path_to_rdf>")
        sys.exit(1)
    analyze_rdf(sys.argv[1]) 