"""
Command-line interface for MariaDB Vector Magics.

This module provides CLI utilities for testing connections, managing secrets, and basic operations.
"""

import argparse
import sys
import logging
import getpass
from typing import Optional

try:
    import mariadb
except ImportError:
    print("ERROR: mariadb package is required. Install with: pip install mariadb")
    sys.exit(1)

from .core.database import DatabaseManager
from .core.secrets import get_secrets_manager, get_secret, set_secret

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_connection() -> None:
    """
    Test MariaDB connection with Vector support.
    
    This function provides a command-line interface to test database connectivity
    and Vector feature availability.
    """
    parser = argparse.ArgumentParser(
        description='Test MariaDB Vector connection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mariadb-vector-test --password mypass
  mariadb-vector-test --host 192.168.1.100 --password mypass --database vectordb
        """
    )
    
    parser.add_argument('--host', default='127.0.0.1', help='MariaDB host address')
    parser.add_argument('--port', type=int, default=3306, help='MariaDB port number')
    parser.add_argument('--user', default='root', help='MariaDB username')
    parser.add_argument('--password', required=True, help='MariaDB password')
    parser.add_argument('--database', default='vectordb', help='Database name')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    db_manager = DatabaseManager()
    
    try:
        print(f"Connecting to MariaDB at {args.host}:{args.port}...")
        
        success = db_manager.connect(
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
            database=args.database
        )
        
        if not success:
            print("✗ Connection failed")
            sys.exit(1)
        
        print("✓ Connection successful!")
        
        # Get version information
        version = db_manager.get_version()
        print(f"✓ MariaDB Version: {version}")
        
        # Test Vector functionality
        vector_support = db_manager.test_vector_support()
        if vector_support:
            print("✓ Vector support: Available")
        else:
            print("✗ Vector support: Not available")
        
        # Test database access
        tables = db_manager.list_tables()
        table_count = len(tables)
        print(f"✓ Database '{args.database}': {table_count} tables found")
        
        if args.verbose and tables:
            print("  Tables:")
            for table in tables:
                print(f"    - {table}")
        
        print("\nAll tests passed! MariaDB Vector Magics should work correctly.")
        
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        logger.error(f"Connection test error: {e}")
        sys.exit(1)
    finally:
        db_manager.disconnect()


def manage_secrets() -> None:
    """
    Manage encrypted secrets for MariaDB Vector Magics.
    """
    parser = argparse.ArgumentParser(
        description='Manage encrypted secrets for MariaDB Vector Magics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mariadb-vector-secrets set GROQ_API_KEY
  mariadb-vector-secrets get GROQ_API_KEY
  mariadb-vector-secrets list
  mariadb-vector-secrets delete GROQ_API_KEY
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Set secret command
    set_parser = subparsers.add_parser('set', help='Set a secret value')
    set_parser.add_argument('name', help='Secret name')
    set_parser.add_argument('value', nargs='?', help='Secret value (will prompt if not provided)')
    
    # Get secret command
    get_parser = subparsers.add_parser('get', help='Get a secret value')
    get_parser.add_argument('name', help='Secret name')
    get_parser.add_argument('--show', action='store_true', help='Show the actual value (use with caution)')
    
    # List secrets command
    list_parser = subparsers.add_parser('list', help='List all secret names')
    
    # Delete secret command
    delete_parser = subparsers.add_parser('delete', help='Delete a secret')
    delete_parser.add_argument('name', help='Secret name')
    
    # Export secrets command
    export_parser = subparsers.add_parser('export', help='Export secrets to file (unencrypted)')
    export_parser.add_argument('file', help='Output file path')
    export_parser.add_argument('--include-env', action='store_true', help='Include environment variables')
    
    # Import secrets command
    import_parser = subparsers.add_parser('import', help='Import secrets from file')
    import_parser.add_argument('file', help='Input file path')
    import_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing secrets')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        secrets_manager = get_secrets_manager()
        
        if args.command == 'set':
            value = args.value
            if not value:
                value = getpass.getpass(f"Enter value for '{args.name}': ")
            
            if not value:
                print("ERROR: Secret value cannot be empty")
                sys.exit(1)
            
            set_secret(args.name, value)
            print(f"✓ Secret '{args.name}' has been set securely")
            
        elif args.command == 'get':
            value = get_secret(args.name)
            if value:
                if args.show:
                    print(f"{args.name}: {value}")
                else:
                    print(f"✓ Secret '{args.name}' found (use --show to display value)")
            else:
                print(f"✗ Secret '{args.name}' not found")
                sys.exit(1)
                
        elif args.command == 'list':
            secrets = secrets_manager.list_secrets()
            if secrets:
                print("Stored secrets:")
                for secret in secrets:
                    print(f"  - {secret}")
            else:
                print("No secrets stored")
                
        elif args.command == 'delete':
            if secrets_manager.delete_secret(args.name):
                print(f"✓ Secret '{args.name}' has been deleted")
            else:
                print(f"✗ Secret '{args.name}' not found")
                sys.exit(1)
                
        elif args.command == 'export':
            secrets_manager.export_secrets(args.file, args.include_env)
            print(f"WARNING: Secrets exported to {args.file} (UNENCRYPTED - handle with care)")
            
        elif args.command == 'import':
            secrets_manager.import_secrets(args.file, args.overwrite)
            print(f"✓ Secrets imported from {args.file}")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        logger.error(f"Secrets management error: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == 'secrets':
        # Remove 'secrets' from argv and call secrets manager
        sys.argv.pop(1)
        manage_secrets()
    else:
        test_connection()


if __name__ == '__main__':
    main()