import networkx as nx

def build_semantic_graph(papers):
    G = nx.MultiDiGraph()
    for idx, paper in enumerate(papers):
        G.add_node(idx, **paper)

    # Tag-based edges
    tag_map = {}
    for i, paper in enumerate(papers):
        for tag in paper['tags']:
            tag_map.setdefault(tag, []).append(i)
    for tag, idxs in tag_map.items():
        for i in idxs:
            for j in idxs:
                if i != j:
                    G.add_edge(i, j, type='shared_tag', tag=tag)

    # Collection-based edges
    collection_map = {}
    for i, paper in enumerate(papers):
        for col in paper['collections']:
            collection_map.setdefault(col, []).append(i)
    for col, idxs in collection_map.items():
        for i in idxs:
            for j in idxs:
                if i != j:
                    G.add_edge(i, j, type='collection', collection=col)

    # Explicit related-item links (if resolvable)
    for i, paper in enumerate(papers):
        for rel in paper['related']:
            for j, other in enumerate(papers):
                if i != j and rel and rel.lower() in other['title'].lower():
                    G.add_edge(i, j, type='related')
    return G

if __name__ == "__main__":
    from biomuse.zotero_parser import parse_zotero_rdf
    papers = parse_zotero_rdf('data/example_zotero_export.rdf')
    G = build_semantic_graph(papers)
    print(nx.info(G))
