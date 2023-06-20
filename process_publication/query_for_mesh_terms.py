##### Script that queries a given disease string for the corresponding MeSh Tree in the NIH
from Bio import Entrez

##### Given the raw results as returned by the MeSh Browser at "https://meshb.nlm.nih.gov",
##### Extract only part of the MeSh Tree Hierarchy using string manipulation methods
def extract_mesh_categories(metadata_string):
    
    ##### If the results do not include a Disease Top-Level Category ([C] - Diseases), return []
    locate_diseases_index = metadata_string.find("Tree Number(s): C")
    if locate_diseases_index != -1:
        mesh_trees = metadata_string.find("All MeSH Categories")
    else:
        return []

    ##### After locating the corresponding MeSh Tree, extract a pre-defined part of it
    ##### Skipping both the Top and Bottom Level Categories
    mesh_categories_string = metadata_string[mesh_trees:]
    
    mesh_categories = []
    for line in mesh_categories_string.split("\n")[2:6]:
        if line.startswith("    "):
            mesh_categories.append(line.strip())
        else:
            break
    
    ##### Return a list of found MeSh terms.
    return mesh_categories


##### Query a given diseases string for the corresponding MeSh term correspondence in NIH
##### The access to The main Entrez web page "http://www.ncbi.nlm.nih.gov/Entrez/" is made through the BioPython library
def get_mesh_tree(query):
    try:
        Entrez.email = "genis.skura@etu.unige.ch"
        handle = Entrez.esearch(db = 'mesh', term = query)
        record = Entrez.read(handle)
        handle.close()
        
        ##### Locate the first found Unique MeSh ID returned by the query
        if len(record["IdList"]) == 0:
            return None
        else:
            mesh_id = record['IdList'][0] if len(record['IdList']) == 1 else record['IdList'][1]
        
        ##### The results of the query are stored in a string
        with Entrez.efetch(db = "mesh", id = mesh_id) as handle:
            metadata_string = handle.read()
        handle.close()

        ##### The corresponding disease MeSh tree is extracted in the custom-built string extraction function declared above
        mesh_categories = extract_mesh_categories(metadata_string)
        return mesh_categories
    except Exception as e:
        # print('MeSh Query Not Found! | Error: {}'.format(str(e)))
        return []