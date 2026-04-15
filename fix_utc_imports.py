#!/usr/bin/env python3
"""Fix UTC imports for Python 3.10 compatibility across freqtrade."""
import os
import re


def fix_utc_imports(filepath):
    """Fix UTC imports in a Python file for Python 3.10 compatibility."""
    with open(filepath, encoding='utf-8') as f:
        content = f.read()
    
    # Check if already fixed
    if 'except ImportError:' in content and 'UTC = timezone.utc' in content:
        return False
    
    original_content = content
    
    # Pattern: from datetime import UTC, ...
    # Replace "from datetime import UTC, X, Y" with "from datetime import X, Y, timezone"
    # and add the fallback
    
    old_imports = []
    new_imports = []
    
    for line in content.split('\n'):
        if line.strip().startswith('from datetime import') and 'UTC' in line:
            old_imports.append(line)
            # Parse the import
            match = re.search(r'from datetime import (.+)', line)
            if match:
                imports_str = match.group(1)
                # Split and clean
                imports = [i.strip() for i in imports_str.split(',') if i.strip() and i.strip() != 'UTC']
                if 'timezone' not in imports:
                    imports.append('timezone')
                new_import = f"from datetime import {', '.join(imports)}"
                new_imports.append(new_import)
    
    # Replace old with new
    for old, new in zip(old_imports, new_imports):
        content = content.replace(old, new, 1)
    
    # Add UTC fallback after datetime imports
    if old_imports and 'except ImportError:' not in content:
        lines = content.split('\n')
        
        # Find the last "from datetime import" or "import datetime" line
        last_datetime_import_idx = -1
        for i in range(len(lines) - 1, -1, -1):
            if 'from datetime import' in lines[i] or (lines[i].strip().startswith('import datetime') and not lines[i].strip().startswith('from')):
                last_datetime_import_idx = i
                break
        
        if last_datetime_import_idx >= 0:
            # Check if fallback already exists nearby
            has_fallback = False
            for i in range(max(0, last_datetime_import_idx - 2), min(len(lines), last_datetime_import_idx + 5)):
                if 'UTC = timezone.utc' in lines[i]:
                    has_fallback = True
                    break
            
            if not has_fallback:
                fallback_code = [
                    "",
                    "# Python 3.11+ compatibility: use timezone.utc instead of UTC",
                    "try:",
                    "    from datetime import UTC",
                    "except ImportError:",
                    "    UTC = timezone.utc"
                ]
                # Insert after the last datetime import
                for code_line in reversed(fallback_code):
                    lines.insert(last_datetime_import_idx + 1, code_line)
                content = '\n'.join(lines)
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    
    return False

# Find all files to fix
freqtrade_path = 'freqtrade/freqtrade'
fixed_count = 0
attempted = 0

for root, dirs, files in os.walk(freqtrade_path):
    # Skip __pycache__ and test directories for now
    dirs[:] = [d for d in dirs if d != '__pycache__']
    
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            try:
                with open(filepath, encoding='utf-8') as f:
                    content = f.read()
                    if 'from datetime import' in content and 'UTC' in content:
                        attempted += 1
                        if fix_utc_imports(filepath):
                            fixed_count += 1
                            print(f"[FIXED] {filepath}")
            except Exception as e:
                print(f"[ERROR] {filepath}: {e}")

print(f"\n✓ Attempted: {attempted} files")
print(f"✓ Fixed: {fixed_count} files")
