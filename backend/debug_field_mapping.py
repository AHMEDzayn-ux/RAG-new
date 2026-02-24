"""Debug script to trace field mapping"""
import sys
sys.path.insert(0, '.')

import json
from pathlib import Path
from services.document_loader import DocumentLoader

# Load JSON
with open("source files/policies.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# Get FUP_VIDEO policy
fup_video = [p for p in data['policies'] if p['policy_id'] == 'FUP_VIDEO'][0]

print("="*80)
print("FUP_VIDEO POLICY OBJECT:")
print(json.dumps(fup_video, indent=2))
print("="*80)

# Initialize loader
loader = DocumentLoader()

# Test field mapping
text_fields, metadata_fields = loader._suggest_field_mapping(fup_video)

print("\nFIELD MAPPING RESULTS:")
print(f"Text fields: {text_fields}")
print(f"Metadata fields: {metadata_fields}")
print("="*80)

# Test conversion to text
text = loader.json_object_to_text(fup_video, text_fields=text_fields)

print("\nCONVERTED TEXT:")
print(text)
print("="*80)

# Check if rules are mentioned
if 'rules' in text.lower():
    print("\n✅ RULES FOUND IN TEXT")
else:
    print("\n❌ RULES NOT FOUND IN TEXT")

if '720p' in text:
    print("✅ '720p' FOUND IN TEXT")
else:
    print("❌ '720p' NOT FOUND IN TEXT")

if '50GB' in text:
    print("✅ '50GB' FOUND IN TEXT")
else:
    print("❌ '50GB' NOT FOUND IN TEXT")
