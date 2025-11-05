#!/usr/bin/env python3
"""
Master Setup Script for Demo System

This script runs all setup steps in the correct order:
1. Verify Neo4j connection
2. Verify knowledge graph pipelines
3. Create demo students (Roma and Moha)
4. Run the Flask application
"""

import sys
import os
import subprocess
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def run_script(script_path, description):
    """Run a Python script and return success status."""
    print_header(description)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.dirname(script_path)),
            capture_output=True,
            text=True,
            timeout=120
        )
        
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"\n✅ {description} - SUCCESS")
            return True
        else:
            print(f"\n❌ {description} - FAILED (exit code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"\n❌ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"\n❌ {description} - ERROR: {e}")
        return False


def check_neo4j():
    """Check if Neo4j is running."""
    print_header("Checking Neo4j Status")
    
    try:
        from services.knowledge_graph.services.dynamic_graph_manager import DynamicGraphManager
        
        graph_manager = DynamicGraphManager()
        
        if graph_manager.neo4j and graph_manager.neo4j.graph:
            result = graph_manager.neo4j.graph.query("RETURN 1 as test")
            if result and result[0]['test'] == 1:
                print("✅ Neo4j is running and accessible")
                return True
        
        print("❌ Neo4j is not accessible")
        print("\nPlease start Neo4j with:")
        print("  cd lab_tutor-1")
        print("  docker-compose up -d")
        return False
        
    except Exception as e:
        print(f"❌ Error checking Neo4j: {e}")
        print("\nPlease ensure Neo4j is running:")
        print("  cd lab_tutor-1")
        print("  docker-compose up -d")
        return False


def main():
    """Run the complete setup process."""
    print("\n" + "🚀 "*20)
    print("KTCD_Aug DEMO SYSTEM SETUP")
    print("🚀 "*20)
    
    # Step 1: Check Neo4j
    if not check_neo4j():
        print("\n❌ SETUP FAILED: Neo4j is not running")
        print("\nPlease start Neo4j first:")
        print("  cd lab_tutor-1")
        print("  docker-compose up -d")
        print("\nThen run this script again.")
        return False
    
    # Step 2: Verify pipelines
    utils_dir = os.path.dirname(os.path.abspath(__file__))
    verify_script = os.path.join(utils_dir, 'verify_and_test_pipelines.py')
    
    if not run_script(verify_script, "Verifying Knowledge Graph Pipelines"):
        print("\n⚠️  Pipeline verification had issues, but continuing...")
    
    # Step 3: Create demo students
    demo_script = os.path.join(utils_dir, 'create_demo_students.py')
    
    if not run_script(demo_script, "Creating Demo Students (Roma & Moha)"):
        print("\n❌ SETUP FAILED: Could not create demo students")
        return False
    
    # Success!
    print_header("SETUP COMPLETE!")
    
    print("✅ All setup steps completed successfully!\n")
    print("Demo Students Created:")
    print("  1. Roma")
    print("     - Email: roma@example.com")
    print("     - Password: roma123")
    print("     - Strengths: NoSQL concepts")
    print("     - Weaknesses: Algorithms")
    print()
    print("  2. Moha")
    print("     - Email: moha@example.com")
    print("     - Password: moha123")
    print("     - Strengths: Algorithms")
    print("     - Weaknesses: NoSQL Theory")
    print()
    print("Next Steps:")
    print("  1. Start the Flask application:")
    print("     python3 nexus_app.py")
    print()
    print("  2. Open your browser to:")
    print("     http://127.0.0.1:8080/student/")
    print()
    print("  3. Login as Roma or Moha to see their unique portfolios!")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

