# 🚨 Critical Fixes Implementation Guide

## Quick Start - Fix Critical Security Issues NOW

This guide provides **copy-paste ready code** to fix the 3 most critical security vulnerabilities in your KTCD_Aug platform.

---

## 🔴 FIX #1: Secure Secret Key (5 minutes)

### Step 1: Create `.env` file
```bash
cd /Users/mohasani/KTCD_Aug
touch .env
echo "SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')" >> .env
echo ".env" >> .gitignore  # Never commit secrets!
```

### Step 2: Install python-dotenv
```bash
pip install python-dotenv
```

### Step 3: Update `nexus_app.py`
Replace line 75 with:
```python
# OLD (line 75):
# app.secret_key = 'bit_tutor_ultra_secure_key_2025'

# NEW:
import os
from dotenv import load_dotenv
import secrets

# Load environment variables
load_dotenv()

# Use environment variable or generate secure random key
app.secret_key = os.environ.get('SECRET_KEY')
if not app.secret_key:
    app.secret_key = secrets.token_hex(32)
    print("⚠️  WARNING: Using generated secret key. Set SECRET_KEY in .env for production!")
```

### Step 4: Verify
```bash
python3 -c "from dotenv import load_dotenv; import os; load_dotenv(); print('✅ Secret key loaded' if os.getenv('SECRET_KEY') else '❌ No secret key')"
```

---

## 🔴 FIX #2: Password Hashing (15 minutes)

### Step 1: Install bcrypt
```bash
pip install bcrypt
```

### Step 2: Create password manager service
```bash
mkdir -p services/auth
touch services/auth/__init__.py
```

Create `services/auth/password_manager.py`:
```python
"""
Password Manager Service
Handles secure password hashing and verification using bcrypt
"""
import bcrypt

class PasswordManager:
    """Secure password hashing and verification"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password using bcrypt with automatic salt generation.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password string
        """
        if not password:
            raise ValueError("Password cannot be empty")
        
        # Generate salt and hash password
        salt = bcrypt.gensalt(rounds=12)  # 12 rounds = good security/performance balance
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            password: Plain text password to verify
            hashed: Hashed password from database
            
        Returns:
            True if password matches, False otherwise
        """
        if not password or not hashed:
            return False
        
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            print(f"Password verification error: {e}")
            return False
```

### Step 3: Update teacher login route
Edit `routes/teacher_routes.py`:
```python
# Add import at top
from services.auth.password_manager import PasswordManager

# Replace the login function (around line 40-65):
@teacher_bp.route('/login', methods=['POST'])
def login():
    """Process teacher login with secure password verification"""
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'email' not in data or 'password' not in data:
            return jsonify({'success': False, 'message': 'Email and password required'}), 400
        
        # Get teacher by email only (don't include password in query!)
        query = """
        MATCH (t:Teacher {email: $email})
        RETURN t.teacher_id as teacher_id,
               t.name as name,
               t.email as email,
               t.password_hash as password_hash
        """
        result = graph_manager.neo4j.graph.query(query, {'email': data['email']})
        
        if result and len(result) > 0:
            teacher = result[0]
            
            # Verify password hash
            if PasswordManager.verify_password(data['password'], teacher['password_hash']):
                # Login successful
                session['teacher_id'] = teacher['teacher_id']
                session['teacher_name'] = teacher['name']
                session['teacher_email'] = teacher['email']
                
                return jsonify({
                    'success': True,
                    'message': 'Login successful',
                    'redirect': url_for('teacher.dashboard')
                })
        
        # Invalid credentials (don't specify which field is wrong!)
        return jsonify({'success': False, 'message': 'Invalid email or password'}), 401
        
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'success': False, 'message': 'An error occurred'}), 500
```

### Step 4: Migrate existing passwords
Create `utilities/migrate_teacher_passwords.py`:
```python
"""
Migrate teacher passwords from plain text to bcrypt hashes
RUN THIS ONCE to update existing teacher accounts
"""
import sys
sys.path.insert(0, '/Users/mohasani/KTCD_Aug')

from services.knowledge_graph.services.dynamic_graph_manager import DynamicGraphManager
from services.auth.password_manager import PasswordManager

def migrate_passwords():
    """Migrate all teacher passwords to hashed format"""
    print("🔐 Starting password migration...")
    
    graph_manager = DynamicGraphManager()
    
    # Get all teachers with plain text passwords
    query = """
    MATCH (t:Teacher)
    WHERE exists(t.password)
    RETURN t.teacher_id as id, t.password as plain_password
    """
    teachers = graph_manager.neo4j.graph.query(query)
    
    print(f"Found {len(teachers)} teachers to migrate")
    
    for teacher in teachers:
        teacher_id = teacher['id']
        plain_password = teacher['plain_password']
        
        # Hash the password
        hashed = PasswordManager.hash_password(plain_password)
        
        # Update in database
        update_query = """
        MATCH (t:Teacher {teacher_id: $id})
        SET t.password_hash = $hash
        REMOVE t.password
        """
        graph_manager.neo4j.graph.query(update_query, {
            'id': teacher_id,
            'hash': hashed
        })
        
        print(f"✅ Migrated password for teacher: {teacher_id}")
    
    print("🎉 Password migration complete!")

if __name__ == '__main__':
    migrate_passwords()
```

Run the migration:
```bash
python3 utilities/migrate_teacher_passwords.py
```

### Step 5: Update student authentication
The student auth service already uses SHA-256 hashing, but let's upgrade it to bcrypt.

Edit `services/auth/student_auth.py` (around line 30):
```python
# Replace SHA-256 with bcrypt
from services.auth.password_manager import PasswordManager

class StudentAuthService:
    def register_student(self, name, email, password, hobbies=None, interests=None):
        """Register new student with hashed password"""
        # ... existing validation code ...
        
        # Hash password with bcrypt (replace SHA-256)
        password_hash = PasswordManager.hash_password(password)
        
        # ... rest of registration code ...
        # Use password_hash instead of hashlib.sha256
    
    def login_student(self, email, password):
        """Login student with password verification"""
        # Get student by email
        query = """
        MATCH (s:Student {email: $email})
        RETURN s
        """
        result = self.graph_manager.neo4j.graph.query(query, {'email': email})
        
        if result and len(result) > 0:
            student = result[0]['s']
            
            # Verify password
            if PasswordManager.verify_password(password, student.get('password_hash', '')):
                return {
                    'success': True,
                    'student_id': student['student_id'],
                    'name': student['name'],
                    'email': student['email']
                }
        
        return {'success': False, 'message': 'Invalid email or password'}
```

---

## 🔴 FIX #3: Prevent Cypher Injection (10 minutes)

### Step 1: Audit all queries
Search for string interpolation in queries:
```bash
cd /Users/mohasani/KTCD_Aug
grep -r "f\".*MATCH" routes/ services/ --include="*.py"
grep -r "\.format(" routes/ services/ --include="*.py" | grep -i "match\|create\|merge"
```

### Step 2: Fix any vulnerable queries
**WRONG** (vulnerable):
```python
# ❌ NEVER DO THIS
query = f"MATCH (s:Student {{student_id: '{student_id}'}}) RETURN s"
query = "MATCH (s:Student {student_id: '%s'}) RETURN s" % student_id
query = "MATCH (s:Student {student_id: '{}'}) RETURN s".format(student_id)
```

**RIGHT** (safe):
```python
# ✅ ALWAYS DO THIS
query = "MATCH (s:Student {student_id: $student_id}) RETURN s"
result = graph_manager.neo4j.graph.query(query, {'student_id': student_id})
```

### Step 3: Create query helper
Create `utilities/query_validator.py`:
```python
"""
Query Validator - Ensures all Cypher queries use parameterization
"""
import re

class QueryValidator:
    """Validate Cypher queries for security"""
    
    @staticmethod
    def is_parameterized(query: str) -> bool:
        """Check if query uses parameters instead of string interpolation"""
        # Check for f-strings, .format(), or % formatting
        dangerous_patterns = [
            r'f["\'].*MATCH',  # f-string
            r'\.format\(',      # .format()
            r'%s',              # % formatting
            r'%d',
            r'\{[^$]'          # {var} instead of {$param}
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return False
        
        return True
    
    @staticmethod
    def validate_query(query: str, params: dict = None):
        """Validate query before execution"""
        if not QueryValidator.is_parameterized(query):
            raise ValueError(f"Query uses string interpolation! Use parameterized queries: {query[:100]}")
        
        # Check that all parameters in query are provided
        param_names = re.findall(r'\$(\w+)', query)
        if params:
            missing = set(param_names) - set(params.keys())
            if missing:
                raise ValueError(f"Missing parameters: {missing}")
        
        return True

# Use in queries:
from utils.query_validator import QueryValidator

def get_student(student_id):
    query = "MATCH (s:Student {student_id: $student_id}) RETURN s"
    params = {'student_id': student_id}
    
    # Validate before executing
    QueryValidator.validate_query(query, params)
    
    return graph_manager.neo4j.graph.query(query, params)
```

---

## ✅ Verification Checklist

After implementing all fixes, verify:

```bash
# 1. Check secret key is from environment
python3 -c "from dotenv import load_dotenv; import os; load_dotenv(); print('✅ Secret key loaded' if os.getenv('SECRET_KEY') else '❌ Failed')"

# 2. Test password hashing
python3 -c "from services.auth.password_manager import PasswordManager; h = PasswordManager.hash_password('test123'); print('✅ Hashing works' if PasswordManager.verify_password('test123', h) else '❌ Failed')"

# 3. Check for vulnerable queries
grep -r "f\".*MATCH" routes/ services/ --include="*.py" && echo "❌ Found vulnerable queries" || echo "✅ No vulnerable queries found"

# 4. Test teacher login
curl -X POST http://127.0.0.1:8080/teacher/login \
  -H "Content-Type: application/json" \
  -d '{"email":"teacher@example.com","password":"teacher123"}'

# 5. Test student login
curl -X POST http://127.0.0.1:8080/student/login \
  -H "Content-Type: application/json" \
  -d '{"email":"roma@example.com","password":"roma123"}'
```

---

## 📋 Summary

**Time Required**: ~30 minutes  
**Difficulty**: Easy to Medium  
**Impact**: CRITICAL - Prevents major security breaches

**What You Fixed**:
1. ✅ Hardcoded secret key → Environment variable
2. ✅ Plain text passwords → Bcrypt hashing
3. ✅ Cypher injection → Parameterized queries

**Next Steps**:
- See `CODE_IMPROVEMENT_ANALYSIS.md` for remaining issues
- Implement rate limiting (HIGH priority)
- Add CSRF protection (HIGH priority)
- Set up proper logging (MEDIUM priority)

---

**Status**: Ready to implement  
**Last Updated**: November 3, 2025

