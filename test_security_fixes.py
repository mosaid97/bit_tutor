#!/usr/bin/env python3
"""
Test Security Fixes
Quick verification that all security fixes are working
"""

print('🔍 Testing Security Fixes...')
print('-' * 60)

# Test 1: Environment variables
from dotenv import load_dotenv
import os
load_dotenv()
secret_key = os.getenv('SECRET_KEY')
result1 = '✅ Loaded' if secret_key else '❌ Missing'
print(f'1. Secret Key: {result1}')

# Test 2: Password Manager
from services.auth.password_manager import PasswordManager
test_pass = 'test123'
hashed = PasswordManager.hash_password(test_pass)
verified = PasswordManager.verify_password(test_pass, hashed)
result2 = '✅ Working' if verified else '❌ Failed'
print(f'2. Password Hashing: {result2}')

# Test 3: Bcrypt format
is_bcrypt = hashed.startswith('$2b$')
result3 = '✅ Correct' if is_bcrypt else '❌ Wrong'
print(f'3. Bcrypt Format: {result3}')

# Test 4: Wrong password rejected
wrong_verified = PasswordManager.verify_password('wrongpass', hashed)
result4 = '✅ Rejected' if not wrong_verified else '❌ Accepted'
print(f'4. Wrong Password: {result4}')

print('-' * 60)

if all([secret_key, verified, is_bcrypt, not wrong_verified]):
    print('✅ All security fixes verified!')
    exit(0)
else:
    print('❌ Some tests failed!')
    exit(1)

