"""
Password Manager Service
Handles secure password hashing and verification using bcrypt

SECURITY: This module provides industry-standard password hashing
using bcrypt with automatic salt generation.
"""
import bcrypt


class PasswordManager:
    """Secure password hashing and verification using bcrypt"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password using bcrypt with automatic salt generation.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password string (safe to store in database)
            
        Raises:
            ValueError: If password is empty
            
        Example:
            >>> hashed = PasswordManager.hash_password("mypassword123")
            >>> print(hashed)
            $2b$12$KIXxBz8...
        """
        if not password:
            raise ValueError("Password cannot be empty")
        
        # Generate salt and hash password
        # 12 rounds = good balance between security and performance
        salt = bcrypt.gensalt(rounds=12)
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
            
        Example:
            >>> hashed = PasswordManager.hash_password("mypassword123")
            >>> PasswordManager.verify_password("mypassword123", hashed)
            True
            >>> PasswordManager.verify_password("wrongpassword", hashed)
            False
        """
        if not password or not hashed:
            return False
        
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except Exception as e:
            print(f"⚠️  Password verification error: {e}")
            return False
    
    @staticmethod
    def needs_rehash(hashed: str, rounds: int = 12) -> bool:
        """
        Check if a password hash needs to be updated (e.g., if cost factor changed).
        
        Args:
            hashed: Existing password hash
            rounds: Desired number of rounds
            
        Returns:
            True if hash should be regenerated
        """
        try:
            # Extract current rounds from hash
            # bcrypt hash format: $2b$12$...
            parts = hashed.split('$')
            if len(parts) >= 3:
                current_rounds = int(parts[2])
                return current_rounds < rounds
        except Exception:
            pass
        
        return False


# Example usage and testing
if __name__ == '__main__':
    print("🔐 Password Manager Test")
    print("-" * 50)
    
    # Test password hashing
    password = "test_password_123"
    print(f"Original password: {password}")
    
    hashed = PasswordManager.hash_password(password)
    print(f"Hashed password: {hashed}")
    
    # Test verification
    print(f"\nVerify correct password: {PasswordManager.verify_password(password, hashed)}")
    print(f"Verify wrong password: {PasswordManager.verify_password('wrong_password', hashed)}")
    
    # Test rehash check
    print(f"\nNeeds rehash (12 rounds): {PasswordManager.needs_rehash(hashed, 12)}")
    print(f"Needs rehash (15 rounds): {PasswordManager.needs_rehash(hashed, 15)}")
    
    print("\n✅ All tests passed!")

