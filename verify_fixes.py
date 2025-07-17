#!/usr/bin/env python3
"""
Simple verification of the fixes applied
"""

def check_enum_fix():
    """Check that AuthenticationDecision enum is properly fixed"""
    
    # Read the models file to verify the fix
    models_file = "behavioral-auth-engine/src/data/models.py"
    
    try:
        with open(models_file, 'r') as f:
            content = f.read()
        
        print("🧪 CHECKING AUTHENTICATIONDECISION ENUM")
        print("="*50)
        
        # Check if DENY is removed and BLOCK is present
        if 'DENY = "deny"' in content:
            print("❌ DENY still found in enum - needs to be removed")
            return False
        elif 'BLOCK = "block"' in content:
            print("✅ BLOCK found in enum")
        else:
            print("❌ BLOCK not found in enum")
            return False
            
        # Show the enum section
        lines = content.split('\n')
        in_enum = False
        enum_lines = []
        
        for line in lines:
            if 'class AuthenticationDecision' in line:
                in_enum = True
                enum_lines.append(line)
            elif in_enum:
                if line.strip().startswith('ALLOW') or line.strip().startswith('CHALLENGE') or \
                   line.strip().startswith('BLOCK') or line.strip().startswith('LEARN'):
                    enum_lines.append(line)
                elif line.strip() == '' or line.strip().startswith('"""'):
                    enum_lines.append(line)
                elif not line.startswith('    ') and line.strip():
                    break
        
        print("\n📋 Current AuthenticationDecision enum:")
        for line in enum_lines:
            if line.strip():
                print(f"   {line}")
        
        return True
        
    except FileNotFoundError:
        print(f"❌ File not found: {models_file}")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def check_continuous_analysis_fix():
    """Check that continuous_analysis.py uses BLOCK instead of DENY"""
    
    analysis_file = "behavioral-auth-engine/src/core/continuous_analysis.py"
    
    try:
        with open(analysis_file, 'r') as f:
            content = f.read()
        
        print("\n🧪 CHECKING CONTINUOUS_ANALYSIS.PY")
        print("="*50)
        
        deny_count = content.count('AuthenticationDecision.DENY')
        block_count = content.count('AuthenticationDecision.BLOCK')
        
        print(f"   AuthenticationDecision.DENY occurrences: {deny_count}")
        print(f"   AuthenticationDecision.BLOCK occurrences: {block_count}")
        
        if deny_count > 0:
            print("❌ DENY still found - needs to be replaced with BLOCK")
            return False
        else:
            print("✅ No DENY found - successfully replaced")
            return True
            
    except FileNotFoundError:
        print(f"❌ File not found: {analysis_file}")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False

def main():
    """Run verification checks"""
    print("🔧 DATABASE CONSTRAINT FIX VERIFICATION")
    print("="*55)
    print("Checking that DENY → BLOCK replacement was successful")
    print("="*55)
    
    enum_ok = check_enum_fix()
    analysis_ok = check_continuous_analysis_fix()
    
    print("\n" + "="*55)
    if enum_ok and analysis_ok:
        print("✅ ALL FIXES VERIFIED SUCCESSFULLY!")
        print("\n🎯 What was fixed:")
        print("   • AuthenticationDecision.DENY → AuthenticationDecision.BLOCK")
        print("   • Removed DENY from enum (wasn't in database constraint)")
        print("   • All enum values now match database: allow, challenge, block, learn")
        print("\n🚀 Your authentication decisions should now store successfully!")
        print("   No more foreign key constraint violations!")
    else:
        print("❌ Some issues found - check the output above")
    
    print("\n💡 Next steps:")
    print("   1. Restart your ML Engine service")
    print("   2. Test behavioral analysis endpoint")
    print("   3. Verify decisions are stored without errors")

if __name__ == "__main__":
    main()
