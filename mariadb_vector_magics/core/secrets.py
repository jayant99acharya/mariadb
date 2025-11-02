"""
Secure secrets management for MariaDB Vector Magics.

This module provides secure handling of API keys and other sensitive configuration.
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from .exceptions import SecretsError

logger = logging.getLogger(__name__)


class SecretsManager:
    """Manages secure storage and retrieval of secrets."""
    
    def __init__(self, secrets_file: Optional[str] = None, master_key: Optional[str] = None):
        """
        Initialize secrets manager.
        
        Args:
            secrets_file: Path to encrypted secrets file
            master_key: Master key for encryption (if None, uses environment variable)
        """
        self.secrets_file = Path(secrets_file) if secrets_file else self._get_default_secrets_file()
        self.master_key = master_key or os.getenv('MARIADB_VECTOR_MASTER_KEY')
        self._cipher_suite: Optional[Fernet] = None
        self._secrets_cache: Dict[str, str] = {}
    
    def _get_default_secrets_file(self) -> Path:
        """Get default secrets file location."""
        # Try user's home directory first
        home_dir = Path.home()
        secrets_dir = home_dir / '.mariadb_vector'
        secrets_dir.mkdir(exist_ok=True)
        return secrets_dir / 'secrets.enc'
    
    def _get_cipher_suite(self) -> Fernet:
        """Get or create cipher suite for encryption/decryption."""
        if self._cipher_suite is not None:
            return self._cipher_suite
        
        if not self.master_key:
            raise SecretsError(
                "Master key not provided. Set MARIADB_VECTOR_MASTER_KEY environment variable "
                "or provide master_key parameter"
            )
        
        try:
            # Derive key from master key
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'mariadb_vector_salt',  # In production, use random salt
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
            self._cipher_suite = Fernet(key)
            return self._cipher_suite
            
        except Exception as e:
            raise SecretsError(f"Failed to initialize encryption: {e}")
    
    def _load_secrets(self) -> Dict[str, str]:
        """Load and decrypt secrets from file."""
        if not self.secrets_file.exists():
            logger.info(f"Secrets file {self.secrets_file} does not exist, starting with empty secrets")
            return {}
        
        try:
            cipher_suite = self._get_cipher_suite()
            
            with open(self.secrets_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = cipher_suite.decrypt(encrypted_data)
            secrets = json.loads(decrypted_data.decode())
            
            logger.debug(f"Loaded {len(secrets)} secrets from {self.secrets_file}")
            return secrets
            
        except Exception as e:
            raise SecretsError(f"Failed to load secrets from {self.secrets_file}: {e}")
    
    def _save_secrets(self, secrets: Dict[str, str]) -> None:
        """Encrypt and save secrets to file."""
        try:
            cipher_suite = self._get_cipher_suite()
            
            # Ensure directory exists
            self.secrets_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Encrypt secrets
            secrets_json = json.dumps(secrets, indent=2)
            encrypted_data = cipher_suite.encrypt(secrets_json.encode())
            
            # Write to file with secure permissions
            with open(self.secrets_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Set secure file permissions (owner read/write only)
            os.chmod(self.secrets_file, 0o600)
            
            logger.debug(f"Saved {len(secrets)} secrets to {self.secrets_file}")
            
        except Exception as e:
            raise SecretsError(f"Failed to save secrets to {self.secrets_file}: {e}")
    
    def get_secret(self, name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret value.
        
        Args:
            name: Secret name
            default: Default value if secret not found
            
        Returns:
            Secret value or default
        """
        # First check environment variables
        env_value = os.getenv(name)
        if env_value:
            logger.debug(f"Retrieved secret '{name}' from environment variable")
            return env_value
        
        # Then check cache
        if name in self._secrets_cache:
            logger.debug(f"Retrieved secret '{name}' from cache")
            return self._secrets_cache[name]
        
        # Finally check encrypted file
        try:
            secrets = self._load_secrets()
            value = secrets.get(name, default)
            
            if value:
                # Cache the value
                self._secrets_cache[name] = value
                logger.debug(f"Retrieved secret '{name}' from encrypted file")
            
            return value
            
        except SecretsError as e:
            logger.error(f"Failed to get secret '{name}': {e}")
            return default
    
    def set_secret(self, name: str, value: str) -> None:
        """
        Set a secret value.
        
        Args:
            name: Secret name
            value: Secret value
        """
        try:
            # Load existing secrets
            secrets = self._load_secrets()
            
            # Update with new value
            secrets[name] = value
            
            # Save back to file
            self._save_secrets(secrets)
            
            # Update cache
            self._secrets_cache[name] = value
            
            logger.info(f"Secret '{name}' has been set")
            
        except Exception as e:
            raise SecretsError(f"Failed to set secret '{name}': {e}")
    
    def delete_secret(self, name: str) -> bool:
        """
        Delete a secret.
        
        Args:
            name: Secret name
            
        Returns:
            True if secret was deleted, False if not found
        """
        try:
            # Load existing secrets
            secrets = self._load_secrets()
            
            if name not in secrets:
                return False
            
            # Remove secret
            del secrets[name]
            
            # Save back to file
            self._save_secrets(secrets)
            
            # Remove from cache
            self._secrets_cache.pop(name, None)
            
            logger.info(f"Secret '{name}' has been deleted")
            return True
            
        except Exception as e:
            raise SecretsError(f"Failed to delete secret '{name}': {e}")
    
    def list_secrets(self) -> list[str]:
        """List all secret names (not values)."""
        try:
            secrets = self._load_secrets()
            return list(secrets.keys())
        except Exception as e:
            raise SecretsError(f"Failed to list secrets: {e}")
    
    def clear_cache(self) -> None:
        """Clear the secrets cache."""
        self._secrets_cache.clear()
        logger.debug("Secrets cache cleared")
    
    def export_secrets(self, output_file: str, include_env: bool = False) -> None:
        """
        Export secrets to a file (unencrypted - use with caution).
        
        Args:
            output_file: Output file path
            include_env: Whether to include environment variables
        """
        try:
            secrets = self._load_secrets()
            
            if include_env:
                # Add environment variables that look like secrets
                for key, value in os.environ.items():
                    if any(keyword in key.upper() for keyword in ['KEY', 'SECRET', 'TOKEN', 'PASSWORD']):
                        secrets[f"ENV_{key}"] = value
            
            with open(output_file, 'w') as f:
                json.dump(secrets, f, indent=2)
            
            logger.warning(f"Secrets exported to {output_file} (UNENCRYPTED)")
            
        except Exception as e:
            raise SecretsError(f"Failed to export secrets: {e}")
    
    def import_secrets(self, input_file: str, overwrite: bool = False) -> None:
        """
        Import secrets from a file.
        
        Args:
            input_file: Input file path
            overwrite: Whether to overwrite existing secrets
        """
        try:
            with open(input_file, 'r') as f:
                new_secrets = json.load(f)
            
            # Load existing secrets
            existing_secrets = self._load_secrets()
            
            # Merge secrets
            for name, value in new_secrets.items():
                if name in existing_secrets and not overwrite:
                    logger.warning(f"Skipping existing secret '{name}' (use overwrite=True to replace)")
                    continue
                existing_secrets[name] = value
            
            # Save merged secrets
            self._save_secrets(existing_secrets)
            
            # Clear cache to force reload
            self.clear_cache()
            
            logger.info(f"Imported secrets from {input_file}")
            
        except Exception as e:
            raise SecretsError(f"Failed to import secrets: {e}")


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get the global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """Get a secret value using the global secrets manager."""
    return get_secrets_manager().get_secret(name, default)


def set_secret(name: str, value: str) -> None:
    """Set a secret value using the global secrets manager."""
    get_secrets_manager().set_secret(name, value)


def delete_secret(name: str) -> bool:
    """Delete a secret using the global secrets manager."""
    return get_secrets_manager().delete_secret(name)


# Convenience function for common secrets
def get_groq_api_key() -> Optional[str]:
    """Get Groq API key from secrets or environment."""
    return get_secret('GROQ_API_KEY')


def set_groq_api_key(api_key: str) -> None:
    """Set Groq API key in secrets."""
    set_secret('GROQ_API_KEY', api_key)