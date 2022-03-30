# TODO: There needs to be some more work to restrict certain
#       atoms to fields, e.g., date->dp, but not date->title.

mapping = {
    # TODO: At the moment, the fields must be arrays, since
    #       the title and the abstract are separate fields.
    "All Fields": ["title", "abstract", "keyword_list", "mesh_heading_list", "publication_type"],
    "Title/Abstract": ["title", "abstract"],
    "tiab": ["title", "abstract"],
    "Title": ["title"],
    "ti": ["title"],
    "Abstract": ["abstract"],
    "ab": ["abstract"],

    # TODO: In reality, there are many kinds of MeSH headings
    #       so in the future, we may wish to expand these fields.
    "MeSH": ["mesh_heading_list"],
    "MeSH Terms": ["mesh_heading_list"],
    "Mesh": ["mesh_heading_list"],
    "sh": ["mesh_heading_list"],
    "Subheading": ["mesh_heading_list"],
    "Pharmacological Action": ["mesh_heading_list"],  # TODO: This is unlikely correct.
    "Supplementary Concept": ["mesh_heading_list"],  # TODO: This is unlikely correct.
    "nm": ["mesh_heading_list"],  # TODO: This is unlikely correct.
    "MAJR": ["mesh_heading_list"],  # TODO: This is unlikely correct.

    "Publication Type": ["publication_type"],
    "pt": ["publication_type"],

    "Keywords": ["keyword_list"],
    "kw": ["keyword_list"],

    # TODO: for now, there is only a single date field,
    #       which corresponds to the publish date in Pubmed.
    "Publication Date": ["date"],
    "dp": ["date"],

    "PMID": ["pmid"],
    "pmid": ["pmid"],

    # TODO: No mapping yet. Empty list means the
    #       term is not included in the query.
    "jour": [],
}
