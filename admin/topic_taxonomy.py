"""
Fixed sample-type taxonomy and LLM prompt template for topic tagging.
Single source of truth: edit SAMPLE_TYPES here and the backfill script +
admin UI pick up the change on next run/load.
"""

# Fixed sample-type taxonomy. Pick short, plain-English labels.
# "methodology / instrumentation" is for papers that are about the XPCS
# technique itself (algorithms, detectors, beamline design) rather than a
# specific sample. Use "unclear" only when the excerpt genuinely does not
# indicate what was measured.
SAMPLE_TYPES = [
    "nanoparticle suspension",
    "colloidal suspension",
    "colloidal gel",
    "colloidal glass",
    "polymer melt",
    "polymer solution",
    "block copolymer",
    "liquid crystal",
    "protein / biomolecule",
    "membrane / lipid",
    "metallic alloy / metallic glass",
    "foam / emulsion",
    "thin film",
    "methodology / instrumentation",
    "other",
    "unclear",
]


def build_tagging_prompt(title: str, authors: str, excerpt: str) -> str:
    types_bulleted = "\n".join(f"  - {t}" for t in SAMPLE_TYPES)
    return f"""You are tagging an XPCS-related scientific paper. Return ONLY a JSON object, no prose.

Choose exactly one sample_type from this fixed list:
{types_bulleted}

Then generate 3 to 6 freeform lowercase topic tags describing the physics,
measurement, and any distinctive methods (e.g. "aging", "yielding",
"speckle statistics", "two-time correlation", "high-pressure cell").
Keep each tag short (1–4 words) and specific — avoid generic tags like
"xpcs" or "x-ray".

Return this schema exactly:
{{"sample_type": "<one of the list above>", "topics": ["tag1", "tag2", ...]}}

Title: {title}
Authors: {authors}

Excerpt (first ~2000 chars):
{excerpt}
"""
