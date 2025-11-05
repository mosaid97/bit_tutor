#!/usr/bin/env python3
"""
Aggregate Lab Tutor Data

This script aggregates all extraction JSON files from lab_tutor/knowledge_graph_builder/batch_output
into a single complete_neo4j_export_no_embeddings.json file that the LabTutorLoader expects.
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def aggregate_batch_output(batch_output_dir: Path, output_file: Path) -> Dict[str, Any]:
    """
    Aggregate all extraction JSON files into a single file.
    
    Args:
        batch_output_dir: Path to batch_output directory
        output_file: Path to output JSON file
        
    Returns:
        Dictionary with aggregated data
    """
    topics = []
    theories = []
    topic_names_seen = set()
    
    print(f"🔍 Scanning {batch_output_dir} for extraction files...")
    
    # Find all extraction JSON files
    extraction_files = list(batch_output_dir.rglob("*_extraction.json"))
    
    print(f"📄 Found {len(extraction_files)} extraction files")
    
    for json_file in extraction_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            topic_name = data.get('topic', 'Unknown Topic')
            
            # Create topic entry (avoid duplicates)
            if topic_name not in topic_names_seen:
                topic_entry = {
                    "name": topic_name,
                    "summary": data.get('summary', ''),
                    "keywords": data.get('keywords', [])
                }
                topics.append(topic_entry)
                topic_names_seen.add(topic_name)
            
            # Create theory entry (each file is a theory/lecture)
            theory_entry = {
                "name": topic_name,  # Theory name same as topic for now
                "topic": topic_name,
                "summary": data.get('summary', ''),
                "keywords": data.get('keywords', []),
                "concepts": data.get('concepts', []),
                "original_text": data.get('original_text', '')
            }
            theories.append(theory_entry)
            
            print(f"  ✅ Processed: {topic_name}")
            
        except Exception as e:
            print(f"  ⚠️  Error processing {json_file.name}: {e}")
            continue
    
    # Create aggregated data structure
    aggregated_data = {
        "topics": topics,
        "theories": theories
    }
    
    # Save to output file
    print(f"\n💾 Saving aggregated data to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(aggregated_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Aggregation complete!")
    print(f"   📊 Topics: {len(topics)}")
    print(f"   📚 Theories: {len(theories)}")
    print(f"   📝 Total concepts: {sum(len(t.get('concepts', [])) for t in theories)}")
    
    return aggregated_data


def main():
    """Main entry point."""
    # Paths
    base_path = Path(__file__).parent.parent
    batch_output_dir = base_path / "lab_tutor" / "knowledge_graph_builder" / "batch_output"
    output_file = base_path / "lab_tutor" / "knowledge_graph_builder" / "complete_neo4j_export_no_embeddings.json"
    
    print("🚀 Lab Tutor Data Aggregation")
    print("=" * 60)
    print(f"📁 Input: {batch_output_dir}")
    print(f"📄 Output: {output_file}")
    print("-" * 60)
    
    # Check if batch_output exists
    if not batch_output_dir.exists():
        print(f"❌ Error: Batch output directory not found: {batch_output_dir}")
        return 1
    
    # Aggregate data
    try:
        aggregated_data = aggregate_batch_output(batch_output_dir, output_file)
        print(f"\n🎉 Success! Data aggregated and saved to:")
        print(f"   {output_file}")
        return 0
    except Exception as e:
        print(f"\n❌ Error during aggregation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

