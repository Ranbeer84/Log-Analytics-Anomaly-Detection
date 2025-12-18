"""
Verify and patch IsolationForestDetector if needed
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'ml'))

print("Checking IsolationForestDetector...")

try:
    from models.isolation_forest import IsolationForestDetector
    import numpy as np
    
    # Create instance
    model = IsolationForestDetector(contamination=0.1)
    
    # Check for required methods
    required_methods = ['fit', 'predict', 'score_samples', 'save', 'load']
    
    print("\nMethod availability:")
    all_present = True
    for method in required_methods:
        has_method = hasattr(model, method) and callable(getattr(model, method))
        status = "✓" if has_method else "✗"
        print(f"  {status} {method}")
        if not has_method:
            all_present = False
    
    if all_present:
        print("\n✅ All required methods present!")
        
        # Test basic functionality
        print("\nTesting basic functionality...")
        X = np.random.randn(100, 10)
        
        # Test fit
        model.fit(X)
        print("  ✓ fit() works")
        
        # Test predict
        predictions = model.predict(X)
        print(f"  ✓ predict() works (found {np.sum(predictions)} anomalies)")
        
        # Test score_samples
        scores = model.score_samples(X)
        print(f"  ✓ score_samples() works (mean score: {np.mean(scores):.3f})")
        
        print("\n🎉 IsolationForestDetector is fully functional!")
        sys.exit(0)
    else:
        print("\n❌ Missing required methods!")
        print("\nThe 'fit' method might be named 'train' in your implementation.")
        print("Checking for 'train' method...")
        
        if hasattr(model, 'train') and callable(getattr(model, 'train')):
            print("  ✓ Found 'train' method")
            print("\n💡 Solution: Add an alias in ml/models/isolation_forest.py:")
            print("""
    def fit(self, X: np.ndarray):
        \"\"\"Alias for train() for scikit-learn compatibility\"\"\"
        return self.train(X)
            """)
        
        sys.exit(1)
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)