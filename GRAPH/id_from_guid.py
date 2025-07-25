import os
import zipfile
import tempfile
import xml.etree.ElementTree as ET

def extract_names_from_dwsim(file_path, target_ids):
    """
    Extract display names for DWSIM objects from their GUIDs in .dwx or .dwxmz files
    
    Args:
        file_path (str): Path to .dwx or .dwxmz file
        target_ids (list): List of GUID-style IDs (e.g., ['MAT-9fa49902-...'])
        
    Returns:
        dict: {id: display_name} mapping for found objects
    """
    # Handle compressed files
    if file_path.lower().endswith('.dwxmz'):
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(file_path, 'r') as z:
                dwx_files = [f for f in z.namelist() if f.lower().endswith('.dwx')]
                if not dwx_files:
                    raise ValueError("No .dwx file found in archive")
                z.extract(dwx_files[0], tmpdir)
                xml_file = os.path.join(tmpdir, dwx_files[0])
                return _parse_dwx_file(xml_file, target_ids)
    
    return _parse_dwx_file(file_path, target_ids)

def _parse_dwx_file(xml_file, target_ids):
    # Normalize target IDs for case-insensitive matching
    target_ids_norm = [id.upper() for id in target_ids]
    results = {}
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Find all simulation objects
    for obj in root.findall('.//SimulationObject'):
        obj_id = obj.get('Name', '')
        obj_id_norm = obj_id.upper()
        
        if obj_id_norm not in target_ids_norm:
            continue
            
        # Try to get display name from GraphicObject tag
        graphic_obj = obj.find('GraphicObject')
        if graphic_obj is not None:
            display_name = graphic_obj.get('Tag')
            if display_name:
                results[obj_id] = display_name
                continue
                
        # Fallback 1: DisplayName property
        for prop in obj.findall('.//Property[@Name="DisplayName"]'):
            if prop.get('Value'):
                results[obj_id] = prop.get('Value')
                break
        else:
            # Fallback 2: Object type + ID fragment
            obj_type = obj.get('ObjectType', 'UNK')
            short_id = obj_id.split('-')[-1][:8]
            results[obj_id] = f"{obj_type}_{short_id}"
            
    return results

# Example usage
if __name__ == "__main__":
    # Path to DWSIM file (.dwx or .dwxmz)
    dwsim_file = "DWSIM_files/fstest.xml"
    
    # List of target IDs to extract
    target_ids = [
        "MAT-9fa49902-b8ef-42f3-819c-b345e73e9124",
        "CO-950b11c6-3d9f-4c32-9cde-71e7e10c2d44",
        "RC-1a7bfe83-7a2f-4685-88a1-8b3f6d9c0b21"
    ]
    
    names = extract_names_from_dwsim(dwsim_file, target_ids)
    
    print("Extracted Names:")
    for obj_id, name in names.items():
        print(f"{obj_id[:8]}...: {name}")