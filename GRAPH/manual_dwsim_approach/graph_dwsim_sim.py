import xml.etree.ElementTree as ET
import networkx as nx
import re

cnt = 1

# --- CONFIG ---
INPUT_XML  = "Examples/heat_exchanger.xml"  # DWSIM simulation file
OUTPUT_GML = "Examples/heat_exchanger.graphml"

# --- PARSE XML ---
tree = ET.parse(INPUT_XML)
root = tree.getroot()

# --- BUILD GRAPH ---
G = nx.DiGraph()
guid_to_tag = {}

# First pass: extract tag names
for go in root.findall(".//GraphicObject"):
    guid = go.findtext("Name")
    tag  = go.findtext("Tag") or guid

    if not tag:
        tag = guid

    print(f"{cnt}: {tag}")
    cnt += 1

    guid_to_tag[guid] = tag
    G.add_node(tag)  # Use tag as node ID

# Second pass: add edges using tag names
for go in root.findall(".//GraphicObject"):
    guid = go.findtext("Name")
    tag = guid_to_tag.get(guid, guid)

    # Input edges: from source to this node
    for conn in go.findall(".//InputConnectors/Connector"):
        src_guid = conn.get("AttachedFromObjID")
        if src_guid and src_guid in guid_to_tag:
            G.add_edge(guid_to_tag[src_guid], tag)

    # Output edges: from this node to destination
    for conn in go.findall(".//OutputConnectors/Connector"):
        dst_guid = conn.get("AttachedToObjID")
        if dst_guid and dst_guid in guid_to_tag:
            G.add_edge(tag, guid_to_tag[dst_guid])

# --- WRITE OUT ---
print("\nFinal Graph:")
for node in G.nodes:
    print(f"Node: {node}")
print()

nx.write_graphml(G, OUTPUT_GML)
print(f"Saved graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges to '{OUTPUT_GML}'")
