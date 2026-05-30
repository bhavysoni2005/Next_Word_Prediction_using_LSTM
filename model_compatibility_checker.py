"""
Model Compatibility Checker
Tests TensorFlow, Keras, and model loading compatibility

Usage:
    python model_compatibility_checker.py
"""

import sys
import os

def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def check_python_version():
    print_section("Python Version Check")
    version = sys.version
    print(f"Python: {version}")
    
    major, minor = sys.version_info[:2]
    if major == 3 and minor == 11:
        print("✅ Python 3.11 - COMPATIBLE")
        return True
    else:
        print(f"❌ Python 3.{minor} - NOT COMPATIBLE (use Python 3.11)")
        return False

def check_tensorflow():
    print_section("TensorFlow Check")
    try:
        import tensorflow as tf
        version = tf.__version__
        print(f"TensorFlow: {version}")
        
        # Check if it's CPU
        if 'cpu' in version.lower():
            print("✅ TensorFlow CPU edition")
        else:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"ℹ️  GPU detected: {len(gpus)} device(s)")
        
        # Check version compatibility
        major, minor = map(int, version.split('.')[:2])
        if major == 2 and minor >= 13:
            print("✅ TensorFlow 2.13+ - COMPATIBLE")
            return True
        else:
            print(f"❌ TensorFlow {major}.{minor} - Consider upgrading")
            return False
            
    except ImportError:
        print("❌ TensorFlow not installed")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_keras():
    print_section("Keras Check")
    try:
        import keras
        print(f"Keras: {keras.__version__}")
        print("✅ Keras installed")
        return True
    except ImportError:
        print("ℹ️  Keras not separately installed (bundled with TensorFlow)")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_dependencies():
    print_section("Required Dependencies")
    
    packages = {
        'numpy': 'Numerical computing',
        'pandas': 'Data processing',
        'matplotlib': 'Visualization',
        'seaborn': 'Statistical visualization',
        'streamlit': 'Web framework',
        'h5py': 'HDF5 file I/O',
        'pickle': 'Serialization (built-in)',
    }
    
    all_ok = True
    for package, description in packages.items():
        try:
            if package == 'pickle':
                print(f"✅ {package:20} - {description} (built-in)")
                continue
                
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✅ {package:20} - {description} ({version})")
        except ImportError:
            print(f"❌ {package:20} - NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_model_files():
    print_section("Model Files Check")
    
    required_files = {
        'lstm_model.h5': 'LSTM model (H5 format)',
        'tokenizer.pkl': 'Tokenizer pickle',
        'max_len.pkl': 'Max length pickle',
    }
    
    all_ok = True
    for filename, description in required_files.items():
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024*1024)
            print(f"✅ {filename:25} - {description} ({size_mb:.2f} MB)")
        else:
            print(f"❌ {filename:25} - NOT FOUND")
            all_ok = False
    
    return all_ok

def check_model_loading():
    print_section("Model Loading Test")
    
    if not os.path.exists('lstm_model.h5'):
        print("⚠️  Skipping - lstm_model.h5 not found")
        return False
    
    try:
        from tensorflow.keras.models import load_model
        print("Loading model...")
        model = load_model('lstm_model.h5')
        print(f"✅ Model loaded successfully")
        print(f"   Architecture: {model.__class__.__name__}")
        
        # Print model summary
        print("\nModel Summary:")
        model.summary()
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def check_tokenizer():
    print_section("Tokenizer Loading Test")
    
    if not os.path.exists('tokenizer.pkl'):
        print("⚠️  Skipping - tokenizer.pkl not found")
        return False
    
    try:
        import pickle
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        print("✅ Tokenizer loaded successfully")
        print(f"   Vocabulary size: {len(tokenizer.word_index)} words")
        return True
    except Exception as e:
        print(f"❌ Tokenizer loading failed: {e}")
        return False

def check_streamlit():
    print_section("Streamlit Configuration")
    
    try:
        import streamlit as st
        print(f"✅ Streamlit {st.__version__} installed")
        
        config_path = '.streamlit/config.toml'
        if os.path.exists(config_path):
            print(f"✅ Streamlit config found: {config_path}")
        else:
            print(f"ℹ️  No config.toml (using defaults)")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("  LSTM Next Word Prediction - Compatibility Checker")
    print("="*60)
    
    results = {
        'Python Version': check_python_version(),
        'TensorFlow': check_tensorflow(),
        'Keras': check_keras(),
        'Dependencies': check_dependencies(),
        'Model Files': check_model_files(),
        'Model Loading': check_model_loading(),
        'Tokenizer Loading': check_tokenizer(),
        'Streamlit': check_streamlit(),
    }
    
    # Summary
    print_section("Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for check, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check:25} {status}")
    
    print(f"\n{passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All systems ready for deployment!")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please fix issues before deploying.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
