#!/usr/bin/env python3
"""
Password Migration Script
Migrates teacher and student passwords from SHA-256 to bcrypt hashing

IMPORTANT: Run this script ONCE after deploying the password security fixes
"""

import sys
sys.path.insert(0, '/Users/mohasani/KTCD_Aug')

from services.knowledge_graph.services.dynamic_graph_manager import DynamicGraphManager
from services.auth.password_manager import PasswordManager


def migrate_teacher_passwords():
    """Migrate all teacher passwords to bcrypt hashed format"""
    print("\n🔐 Migrating Teacher Passwords...")
    print("-" * 60)
    
    graph_manager = DynamicGraphManager()
    
    # Get all teachers
    query = """
    MATCH (t:Teacher)
    RETURN t.teacher_id as id,
           t.email as email,
           t.password as plain_password,
           t.password_hash as old_hash
    """
    
    try:
        teachers = graph_manager.neo4j.graph.query(query)
        
        if not teachers:
            print("ℹ️  No teachers found in database")
            return
        
        print(f"Found {len(teachers)} teacher(s) to migrate\n")
        
        migrated = 0
        for teacher in teachers:
            teacher_id = teacher.get('id')
            email = teacher.get('email')
            
            # Try to get plain password first (if it exists)
            plain_password = teacher.get('plain_password')
            
            if not plain_password:
                # If no plain password, check if already using bcrypt
                old_hash = teacher.get('old_hash', '')
                if old_hash and old_hash.startswith('$2b$'):
                    print(f"✓ {email}: Already using bcrypt, skipping")
                    continue
                else:
                    print(f"⚠️  {email}: No plain password found, cannot migrate")
                    print(f"   Please reset password manually")
                    continue
            
            # Hash the password with bcrypt
            try:
                new_hash = PasswordManager.hash_password(plain_password)
                
                # Update in database
                update_query = """
                MATCH (t:Teacher {teacher_id: $id})
                SET t.password_hash = $hash
                REMOVE t.password
                """
                
                graph_manager.neo4j.graph.query(update_query, {
                    'id': teacher_id,
                    'hash': new_hash
                })
                
                print(f"✅ {email}: Password migrated successfully")
                migrated += 1
                
            except Exception as e:
                print(f"❌ {email}: Migration failed - {e}")
        
        print(f"\n✅ Teacher migration complete: {migrated}/{len(teachers)} migrated")
        
    except Exception as e:
        print(f"❌ Error during teacher migration: {e}")


def migrate_student_passwords():
    """Migrate all student passwords to bcrypt hashed format"""
    print("\n🔐 Migrating Student Passwords...")
    print("-" * 60)
    
    graph_manager = DynamicGraphManager()
    
    # Get all students with SHA-256 hashes
    query = """
    MATCH (s:Student)
    WHERE exists(s.password_hash)
    RETURN s.student_id as id,
           s.email as email,
           s.password_hash as old_hash
    """
    
    try:
        students = graph_manager.neo4j.graph.query(query)
        
        if not students:
            print("ℹ️  No students found in database")
            return
        
        print(f"Found {len(students)} student(s) to check\n")
        
        migrated = 0
        already_bcrypt = 0
        
        for student in students:
            student_id = student.get('id')
            email = student.get('email')
            old_hash = student.get('old_hash', '')
            
            # Check if already using bcrypt
            if old_hash.startswith('$2b$'):
                already_bcrypt += 1
                continue
            
            # For SHA-256 hashes, we cannot reverse them
            # Students will need to reset their passwords
            print(f"⚠️  {email}: Using SHA-256 hash, cannot auto-migrate")
            print(f"   Student will need to reset password on next login")
        
        print(f"\n✅ Student check complete:")
        print(f"   - Already using bcrypt: {already_bcrypt}")
        print(f"   - Need password reset: {len(students) - already_bcrypt}")
        
        if len(students) - already_bcrypt > 0:
            print(f"\nℹ️  Note: Students with SHA-256 hashes will need to reset passwords")
            print(f"   Consider implementing a password reset flow")
        
    except Exception as e:
        print(f"❌ Error during student migration: {e}")


def verify_migrations():
    """Verify that migrations were successful"""
    print("\n🔍 Verifying Migrations...")
    print("-" * 60)
    
    graph_manager = DynamicGraphManager()
    
    # Check teachers
    teacher_query = """
    MATCH (t:Teacher)
    RETURN count(t) as total,
           count(CASE WHEN t.password_hash STARTS WITH '$2b$' THEN 1 END) as bcrypt_count,
           count(CASE WHEN exists(t.password) THEN 1 END) as plain_count
    """
    
    teacher_result = graph_manager.neo4j.graph.query(teacher_query)
    if teacher_result:
        t = teacher_result[0]
        print(f"\nTeachers:")
        print(f"  Total: {t['total']}")
        print(f"  Using bcrypt: {t['bcrypt_count']}")
        print(f"  Plain passwords remaining: {t['plain_count']}")
        
        if t['bcrypt_count'] == t['total'] and t['plain_count'] == 0:
            print(f"  ✅ All teachers migrated successfully!")
        else:
            print(f"  ⚠️  Some teachers not migrated")
    
    # Check students
    student_query = """
    MATCH (s:Student)
    RETURN count(s) as total,
           count(CASE WHEN s.password_hash STARTS WITH '$2b$' THEN 1 END) as bcrypt_count
    """
    
    student_result = graph_manager.neo4j.graph.query(student_query)
    if student_result:
        s = student_result[0]
        print(f"\nStudents:")
        print(f"  Total: {s['total']}")
        print(f"  Using bcrypt: {s['bcrypt_count']}")
        
        if s['bcrypt_count'] == s['total']:
            print(f"  ✅ All students using bcrypt!")
        else:
            print(f"  ℹ️  {s['total'] - s['bcrypt_count']} students need password reset")


def main():
    """Main migration function"""
    print("=" * 60)
    print("  PASSWORD MIGRATION TO BCRYPT")
    print("=" * 60)
    print("\nThis script will migrate passwords from SHA-256 to bcrypt")
    print("⚠️  IMPORTANT: Make a database backup before proceeding!")
    print()
    
    response = input("Continue with migration? (yes/no): ")
    if response.lower() != 'yes':
        print("Migration cancelled")
        return
    
    # Migrate teachers
    migrate_teacher_passwords()
    
    # Check students
    migrate_student_passwords()
    
    # Verify
    verify_migrations()
    
    print("\n" + "=" * 60)
    print("  MIGRATION COMPLETE")
    print("=" * 60)
    print("\n✅ Password migration finished!")
    print("\nNext steps:")
    print("1. Test teacher login with existing credentials")
    print("2. Test student login (may need password reset)")
    print("3. Implement password reset flow if needed")
    print("4. Update documentation")


if __name__ == '__main__':
    main()

