"""
Fix Phase 4 issues automatically
"""
import sys
from pathlib import Path

print("="*70)
print("PHASE 4 - AUTOMATIC FIX SCRIPT")
print("="*70)
print()

issues_fixed = 0
issues_found = 0

# Issue 1: Check IsolationForestDetector
print("[1/2] Checking IsolationForestDetector...")
iso_forest_path = Path('ml/models/isolation_forest.py')

if iso_forest_path.exists():
    content = iso_forest_path.read_text()
    
    # Check if fit method exists
    if 'def fit(self, X:' not in content and 'def fit(self,' not in content:
        issues_found += 1
        print("  ⚠️  Missing 'fit' method")
        
        # Check if train method exists
        if 'def train(self, X:' in content or 'def train(self,' in content:
            print("  💡 Found 'train' method - adding 'fit' alias...")
            
            # Find the train method and add fit alias after it
            lines = content.split('\n')
            new_lines = []
            train_method_found = False
            
            for i, line in enumerate(lines):
                new_lines.append(line)
                
                # Look for end of train method
                if not train_method_found and 'def train(self, X' in line:
                    train_method_found = True
                
                # Add fit method after train method ends
                if train_method_found and line.strip() and not line.strip().startswith('#'):
                    if i + 1 < len(lines):
                        next_line = lines[i + 1]
                        # Check if next line is a new method or class
                        if next_line.strip().startswith('def ') and 'def train' not in next_line:
                            # Insert fit method here
                            new_lines.append('')
                            new_lines.append('    def fit(self, X: np.ndarray):')
                            new_lines.append('        """')
                            new_lines.append('        Fit the model (alias for train)')
                            new_lines.append('        ')
                            new_lines.append('        Args:')
                            new_lines.append('            X: Training data')
                            new_lines.append('        """')
                            new_lines.append('        return self.train(X)')
                            train_method_found = False
                            issues_fixed += 1
                            print("  ✓ Added 'fit' method")
            
            # Write back
            iso_forest_path.write_text('\n'.join(new_lines))
        else:
            print("  ❌ Neither 'fit' nor 'train' method found!")
            print("     Please check ml/models/isolation_forest.py manually")
    else:
        print("  ✓ 'fit' method exists")
else:
    print("  ❌ File not found: ml/models/isolation_forest.py")

# Issue 2: Check evaluation imports
print("\n[2/2] Checking evaluation module imports...")

# Check metrics.py
metrics_path = Path('ml/evaluation/metrics.py')
if metrics_path.exists():
    content = metrics_path.read_text()
    
    if 'import pandas as pd' not in content:
        issues_found += 1
        print("  ⚠️  Missing 'import pandas as pd' in metrics.py")
        
        # Add import
        lines = content.split('\n')
        new_lines = []
        import_added = False
        
        for line in lines:
            if not import_added and line.startswith('import numpy'):
                new_lines.append(line)
                new_lines.append('import pandas as pd')
                import_added = True
                issues_fixed += 1
                print("  ✓ Added pandas import to metrics.py")
            else:
                new_lines.append(line)
        
        metrics_path.write_text('\n'.join(new_lines))
    else:
        print("  ✓ Pandas import exists in metrics.py")
else:
    print("  ❌ File not found: ml/evaluation/metrics.py")

# Check visualization.py
viz_path = Path('ml/evaluation/visualization.py')
if viz_path.exists():
    content = viz_path.read_text()
    
    if 'import pandas as pd' not in content:
        issues_found += 1
        print("  ⚠️  Missing 'import pandas as pd' in visualization.py")
        
        # Add import
        lines = content.split('\n')
        new_lines = []
        import_added = False
        
        for line in lines:
            if not import_added and line.startswith('import numpy'):
                new_lines.append(line)
                new_lines.append('import pandas as pd')
                import_added = True
                issues_fixed += 1
                print("  ✓ Added pandas import to visualization.py")
            else:
                new_lines.append(line)
        
        viz_path.write_text('\n'.join(new_lines))
    else:
        print("  ✓ Pandas import exists in visualization.py")
else:
    print("  ❌ File not found: ml/evaluation/visualization.py")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Issues found: {issues_found}")
print(f"Issues fixed: {issues_fixed}")

if issues_found == 0:
    print("\n✅ No issues found! Phase 4 is ready.")
elif issues_fixed == issues_found:
    print("\n✅ All issues fixed! Please run tests again:")
    print("   python scripts/test_phase4.py")
else:
    print(f"\n⚠️  {issues_found - issues_fixed} issue(s) need manual fixing")
    print("   Please check the messages above")

print()