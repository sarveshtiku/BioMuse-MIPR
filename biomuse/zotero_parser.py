import xml.etree.ElementTree as ET

def parse_zotero_rdf(file_path):
    ns = {
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'dc': 'http://purl.org/dc/elements/1.1/',
        'z': 'http://www.zotero.org/namespaces/export#',
    }
    tree = ET.parse(file_path)
    root = tree.getroot()
    items = []
    for desc in root.findall('rdf:Description', ns):
        title = desc.find('dc:title', ns)
        abstract = desc.find('z:abstractNote', ns)
        tags = [t.text for t in desc.findall('z:tag', ns)]
        notes = desc.find('z:note', ns)
        collections = [c.text for c in desc.findall('z:collection', ns)]
        related = [r.text for r in desc.findall('z:relatedItem', ns)]
        item = {
            'title': title.text if title is not None else '',
            'abstract': abstract.text if abstract is not None else '',
            'tags': tags,
            'notes': notes.text if notes is not None else '',
            'collections': collections,
            'related': related,
        }
        items.append(item)
    return items

if __name__ == "__main__":
    import json
    parsed = parse_zotero_rdf('data/example_zotero_export.rdf')
    print(json.dumps(parsed, indent=2))
