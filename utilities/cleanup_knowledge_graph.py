#!/usr/bin/env python3
"""
Clean up the knowledge graph by removing unused nodes and duplicate data
"""

from neo4j import GraphDatabase

# Neo4j connection
driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'ktcd_password123'))

def cleanup_duplicate_students():
    """Remove duplicate Roma student (student_c8429a29)"""
    with driver.session() as session:
        print("=" * 80)
        print("CLEANING UP DUPLICATE STUDENTS")
        print("=" * 80)
        
        # Delete the duplicate Roma student
        delete_query = """
        MATCH (s:Student {student_id: 'student_c8429a29'})
        OPTIONAL MATCH (s)-[r]-()
        DELETE r, s
        RETURN count(s) as deleted
        """
        result = session.run(delete_query)
        deleted = result.single()['deleted']
        print(f"✅ Deleted duplicate Roma student: {deleted} node(s)")

def cleanup_unused_hobby_nodes():
    """Remove unused Hobby nodes (not connected to anything)"""
    with driver.session() as session:
        print("\n" + "=" * 80)
        print("CLEANING UP UNUSED HOBBY NODES")
        print("=" * 80)
        
        # Delete isolated Hobby nodes
        delete_query = """
        MATCH (h:Hobby)
        WHERE NOT (h)--()
        DELETE h
        RETURN count(h) as deleted
        """
        result = session.run(delete_query)
        deleted = result.single()['deleted']
        print(f"✅ Deleted unused Hobby nodes: {deleted} node(s)")

def cleanup_unused_classes():
    """Check for unused classes but keep them for future use"""
    with driver.session() as session:
        print("\n" + "=" * 80)
        print("CHECKING UNUSED CLASSES")
        print("=" * 80)
        
        # Check which classes are in use
        check_query = """
        MATCH (c:Class)
        OPTIONAL MATCH (c)-[:INCLUDES_TOPIC]->(t:Topic)
        OPTIONAL MATCH (s:Student)-[:REGISTERED_IN]->(c)
        RETURN c.name as class_name,
               c.class_id as class_id,
               count(DISTINCT t) as topics,
               count(DISTINCT s) as students
        """
        result = session.run(check_query)
        
        for record in result:
            status = "✅ IN USE" if record['students'] > 0 or record['topics'] > 0 else "⚠️  UNUSED"
            print(f"{status}: {record['class_name']}")
            print(f"  ID: {record['class_id']}")
            print(f"  Topics: {record['topics']}, Students: {record['students']}")
            print()

def cleanup_orphaned_nodes():
    """Find and optionally remove nodes with no relationships"""
    with driver.session() as session:
        print("\n" + "=" * 80)
        print("CHECKING FOR ORPHANED NODES")
        print("=" * 80)
        
        # Find orphaned nodes (excluding StudentScore which are standalone)
        check_query = """
        MATCH (n)
        WHERE NOT (n)--()
        AND NOT n:StudentScore
        RETURN labels(n)[0] as label, count(n) as count
        ORDER BY count DESC
        """
        result = session.run(check_query)
        
        orphaned = list(result)
        if orphaned:
            print("⚠️  Found orphaned nodes:")
            for record in orphaned:
                print(f"  {record['label']}: {record['count']}")
        else:
            print("✅ No orphaned nodes found (excluding StudentScore)")

def verify_graph_integrity():
    """Verify the graph structure is intact"""
    with driver.session() as session:
        print("\n" + "=" * 80)
        print("VERIFYING GRAPH INTEGRITY")
        print("=" * 80)
        
        # Check critical relationships
        checks = [
            ("Class → Topic", "MATCH (c:Class)-[:INCLUDES_TOPIC]->(t:Topic) RETURN count(*) as count"),
            ("Topic → Concept", "MATCH (t:Topic)-[:INCLUDES_CONCEPT]->(c:Concept) RETURN count(*) as count"),
            ("Topic → Theory", "MATCH (t:Topic)-[:HAS_THEORY]->(th:Theory) RETURN count(*) as count"),
            ("Theory → Video", "MATCH (th:Theory)-[:EXPLAINED_BY]->(v:Video) RETURN count(*) as count"),
            ("Lab → Topic", "MATCH (l:Lab)-[:PRACTICES]->(t:Topic) RETURN count(*) as count"),
            ("Quiz → Topic", "MATCH (q:Quiz)-[:TESTS]->(t:Topic) RETURN count(*) as count"),
            ("Student → Class", "MATCH (s:Student)-[:REGISTERED_IN]->(c:Class) RETURN count(*) as count"),
            ("Student → Topic", "MATCH (s:Student)-[:ENROLLED_IN]->(t:Topic) RETURN count(*) as count"),
            ("Student → Concept", "MATCH (s:Student)-[:KNOWS]->(c:Concept) RETURN count(*) as count"),
        ]
        
        all_good = True
        for name, query in checks:
            result = session.run(query)
            count = result.single()['count']
            status = "✅" if count > 0 else "❌"
            print(f"{status} {name}: {count}")
            if count == 0:
                all_good = False
        
        if all_good:
            print("\n✅ All critical relationships intact!")
        else:
            print("\n⚠️  Some relationships are missing!")

def get_final_statistics():
    """Get final statistics after cleanup"""
    with driver.session() as session:
        print("\n" + "=" * 80)
        print("FINAL STATISTICS")
        print("=" * 80)
        
        # Node counts
        node_query = """
        MATCH (n)
        RETURN labels(n)[0] as label, count(n) as count
        ORDER BY count DESC
        """
        result = session.run(node_query)
        
        print("\n📊 NODE COUNTS:")
        total_nodes = 0
        for record in result:
            count = record['count']
            total_nodes += count
            print(f"  {record['label']:30s}: {count:5d}")
        print(f"  {'TOTAL':30s}: {total_nodes:5d}")
        
        # Relationship counts
        rel_query = """
        MATCH ()-[r]->()
        RETURN type(r) as type, count(r) as count
        ORDER BY count DESC
        """
        result = session.run(rel_query)
        
        print("\n🔗 RELATIONSHIP COUNTS:")
        total_rels = 0
        for record in result:
            count = record['count']
            total_rels += count
            print(f"  {record['type']:30s}: {count:5d}")
        print(f"  {'TOTAL':30s}: {total_rels:5d}")

if __name__ == "__main__":
    try:
        print("\n🧹 STARTING KNOWLEDGE GRAPH CLEANUP\n")
        
        # Run cleanup operations
        cleanup_duplicate_students()
        cleanup_unused_hobby_nodes()
        cleanup_unused_classes()
        cleanup_orphaned_nodes()
        
        # Verify integrity
        verify_graph_integrity()
        
        # Show final stats
        get_final_statistics()
        
        print("\n" + "=" * 80)
        print("✅ CLEANUP COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()

