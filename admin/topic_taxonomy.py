"""
Fixed sample-type taxonomy and LLM prompt template for topic tagging.
Single source of truth: edit SAMPLE_TYPES here and the backfill script +
admin UI pick up the change on next run/load.
"""

# Fixed sample-type taxonomy. Pick short, plain-English labels.
# "methodology / instrumentation" is for papers that are about the XPCS
# technique itself (algorithms, detectors, beamline design) rather than a
# specific sample. Use "unclear" only when the excerpt genuinely does not
# reveal the sample type.
SAMPLE_TYPES = [
    "nanoparticle suspension",
    "colloidal suspension",
    "colloidal gel",
    "colloidal glass",
    "polymer melt / solution",
    "block copolymer",
    "liquid crystal",
    "protein / biomolecule",
    "membrane / lipid",
    "metallic alloy / solid",
    "foam / emulsion",
    "thin film",
    "methodology / instrumentation",
    "other",
    "unclear",
]


TAG_PROMPT_TMPL = (
    "You are cataloguing scientific papers about X-ray Photon Correlation "
    "Spectroscopy (XPCS) for the beamline scientists at APS 8-ID.\n\n"
    "Given the title and excerpt below, classify the paper on two axes.\n\n"
    "1. sample_type: pick EXACTLY ONE label from this list that best describes "
    "the sample or system studied. Use 'methodology / instrumentation' for "
    "papers about XPCS itself rather than a specific sample. Use 'unclear' "
    "only when the excerpt truly does not reveal the sample.\n"
    "Allowed labels: {sample_types}\n\n"
    "2. topics: 3 to 8 short freeform lowercase tags (1-3 words each) "
    "describing the paper. Include physical phenomena (e.g., 'aging', "
    "'gelation', 'yielding', 'glass transition'), technique variants "
    "(e.g., 'rheo-xpcs', 'coherent diffraction', 'speckle statistics'), "
    "and system properties (e.g., 'hard spheres', 'attractive interactions', "
    "'shear flow').\n\n"
    "TITLE: {title}\n\n"
    "EXCERPT: {excerpt}\n\n"
    "Respond with ONLY a raw JSON object, no markdown or code fences. Example:\n"
    '{{"sample_type": "colloidal gel", "topics": ["aging", "gelation", '
    '"attractive interactions", "compressed exponential"]}}'
)
