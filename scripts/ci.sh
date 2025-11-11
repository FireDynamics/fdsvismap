#!/bin/bash

# Exit on first error, set -e would do this but we want to run all checks
set +e
failure=0

location="$(cd "$(dirname "${0}")"; pwd -P)"
root="$(cd "$(dirname "${location}/../..")"; pwd -P)"

echo "🔍 Running local checks..."
echo "=========================================================================="

# Pre-commit checks (linting, formatting, types)
echo "📋 Running pre-commit checks..."
pre-commit run --all-files
if [ $? -ne 0 ]; then
    failure=1
    echo "❌ Pre-commit checks failed"
else
    echo "✅ Pre-commit checks passed"
fi

echo "=========================================================================="

# Optional: Run tests locally too (recommended before push)
echo "🧪 Running tests..."
uv run pytest tests/ -v --tb=short
if [ $? -ne 0 ]; then
    failure=1
    echo "❌ Tests failed"
else
    echo "✅ Tests passed"
fi

echo "=========================================================================="

if [ $failure -eq 1 ]; then
    echo "❌ Some checks failed"
    exit 1
else
    echo "✅ All checks passed!"
    exit 0
fi
