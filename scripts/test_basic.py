#!/usr/bin/env python3
"""
Basic BioMuse Structure Test

This script tests the basic structure and imports of the BioMuse framework
without requiring external dependencies.
"""

import sys
import os

# Add the biomuse package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing BioMuse module imports...")
    
    try:
        # Test core modules
        from biomuse import zotero_parser
        print("✅ zotero_parser imported successfully")
        
        from biomuse import utils
        print("✅ utils imported successfully")
        
        # Test configuration loading
        from biomuse.utils import load_config
        print("✅ load_config imported successfully")
        
        print("\n✅ All basic imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_config_structure():
    """Test configuration file structure."""
    print("\nTesting configuration structure...")
    
    try:
        config_path = '../configs/task_config.yaml'
        if os.path.exists(config_path):
            print("✅ Configuration file exists")
            
            # Try to read the file
            with open(config_path, 'r') as f:
                content = f.read()
                if 'models:' in content and 'tasks:' in content:
                    print("✅ Configuration file has expected structure")
                    return True
                else:
                    print("❌ Configuration file missing expected sections")
                    return False
        else:
            print("❌ Configuration file not found")
            return False
            
    except Exception as e:
        print(f"❌ Error reading configuration: {e}")
        return False

def test_data_structure():
    """Test data directory structure."""
    print("\nTesting data structure...")
    
    data_dir = '../data'
    if os.path.exists(data_dir):
        print("✅ Data directory exists")
        
        # Check for example file
        example_file = os.path.join(data_dir, 'example_zotero_export.rdf')
        if os.path.exists(example_file):
            print("✅ Example Zotero export file exists")
            return True
        else:
            print("⚠️  Example Zotero export file not found (this is expected for new installations)")
            return True
    else:
        print("❌ Data directory not found")
        return False

def main():
    """Run all basic tests."""
    print("🚀 BioMuse Basic Structure Test")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration Structure", test_config_structure),
        ("Data Structure", test_data_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All basic tests passed! BioMuse structure is correct.")
        print("\n🚀 Next steps:")
        print("   • Install dependencies: pip install -r requirements.txt")
        print("   • Set up API keys for model evaluation")
        print("   • Run demo: python scripts/demo.py")
        print("   • Run full benchmark: python scripts/run_benchmark.py")
    else:
        print("❌ Some tests failed. Please check the structure and dependencies.")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 