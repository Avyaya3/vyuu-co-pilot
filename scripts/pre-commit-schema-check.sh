#!/bin/bash

# Pre-commit hook to auto-generate intent schemas when YAML config changes
# Add this to .git/hooks/pre-commit for automatic execution

set -e

echo "🔍 Checking for intent configuration changes..."

# Check if YAML config files are staged
yaml_changed=$(git diff --cached --name-only | grep -E "config/intent_parameters.*\.ya?ml$" | wc -l)

if [ "$yaml_changed" -gt 0 ]; then
    echo "📝 Intent configuration changes detected, regenerating schemas..."
    
    # Run schema generation
    python scripts/generate_intent_schemas.py
    
    # Check if generated file was updated
    if [ -f "src/schemas/generated_intent_schemas.py" ]; then
        # Add generated file to staging if it was modified
        git add src/schemas/generated_intent_schemas.py
        
        echo "✅ Intent schemas updated and staged"
        echo ""
        echo "📋 Changes made:"
        echo "  - Regenerated src/schemas/generated_intent_schemas.py"
        echo "  - Added updated schemas to commit"
        echo ""
        echo "⚠️  Please verify the generated schemas look correct!"
    else
        echo "❌ Schema generation failed!"
        exit 1
    fi
else
    echo "✅ No intent configuration changes detected"
fi

# Validate that generated schemas can be imported
echo "🔬 Validating generated schemas..."
python -c "
try:
    from src.schemas.generated_intent_schemas import DataFetchParams, IntentClassificationResult
    print('✅ Schema validation passed')
except ImportError as e:
    print(f'❌ Schema validation failed: {e}')
    exit(1)
" || exit 1

echo "🎉 Pre-commit checks completed successfully!" 