# Contributing to KTCD_Aug

Thank you for your interest in contributing to KTCD_Aug! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

---

## 📜 Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

### Our Standards

- ✅ Be respectful and inclusive
- ✅ Welcome newcomers and help them learn
- ✅ Focus on constructive feedback
- ✅ Accept responsibility for mistakes
- ❌ No harassment, trolling, or discriminatory behavior

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- Git
- Basic knowledge of Flask, PyTorch, and Neo4j

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/KTCD_Aug.git
cd KTCD_Aug

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/KTCD_Aug.git
```

---

## 💻 Development Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 4. Start Services

```bash
# Start Neo4j
docker-compose up -d neo4j

# Start AI models server
python ai_models_server.py

# Start web application
python nexus_app.py
```

---

## 🤝 How to Contribute

### Types of Contributions

We welcome:

- 🐛 **Bug fixes**
- ✨ **New features**
- 📝 **Documentation improvements**
- 🧪 **Tests**
- 🎨 **UI/UX enhancements**
- ⚡ **Performance optimizations**

### Before You Start

1. **Check existing issues** - Someone might already be working on it
2. **Open an issue** - Discuss your idea before implementing
3. **Get feedback** - Ensure your approach aligns with project goals

---

## 📏 Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

```python
# Good
def calculate_mastery_level(student_id: str, concept: str) -> float:
    """
    Calculate mastery level for a student on a specific concept.
    
    Args:
        student_id: Unique student identifier
        concept: Concept name
        
    Returns:
        Mastery level between 0 and 1
    """
    # Implementation
    pass

# Bad
def calc(s,c):
    # No docstring, unclear names
    pass
```

### Key Principles

- ✅ **Clear naming**: Use descriptive variable and function names
- ✅ **Type hints**: Add type annotations for function parameters and returns
- ✅ **Docstrings**: Document all public functions and classes
- ✅ **Comments**: Explain complex logic
- ✅ **DRY**: Don't Repeat Yourself
- ✅ **SOLID**: Follow SOLID principles

### File Organization

```python
# Standard import order:
# 1. Standard library
import os
import sys
from datetime import datetime

# 2. Third-party packages
import torch
import numpy as np
from flask import Flask, request

# 3. Local imports
from services.knowledge_tracing import SQKTService
from utilities.helpers import calculate_score
```

### Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case()`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: `_leading_underscore`

---

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_knowledge_tracing.py

# Run with coverage
pytest --cov=services --cov-report=html
```

### Writing Tests

```python
import pytest
from services.knowledge_tracing import SQKTService

def test_predict_performance():
    """Test performance prediction"""
    service = SQKTService()
    prediction = service.predict_next_performance('student_123', 'question_456')
    
    assert 0 <= prediction <= 1
    assert isinstance(prediction, float)

def test_invalid_student_id():
    """Test error handling for invalid student ID"""
    service = SQKTService()
    
    with pytest.raises(ValueError):
        service.predict_next_performance('', 'question_456')
```

### Test Coverage

- Aim for **80%+ code coverage**
- Test edge cases and error conditions
- Include integration tests for critical paths

---

## 🔄 Pull Request Process

### 1. Create a Branch

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clean, documented code
- Follow coding standards
- Add tests for new functionality
- Update documentation

### 3. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add personalized lab generation

- Implement PersonalizedLabGenerator class
- Add concept-level mastery integration
- Include hobby-based personalization
- Add comprehensive tests

Closes #123"
```

### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

### 4. Push Changes

```bash
git push origin feature/your-feature-name
```

### 5. Open Pull Request

1. Go to GitHub and open a Pull Request
2. Fill out the PR template
3. Link related issues
4. Request review from maintainers

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] Commit messages are clear
- [ ] PR description explains changes

---

## 📝 Documentation

### Update Documentation

When adding features, update:

- **README.md** - If user-facing changes
- **Docstrings** - For all new functions/classes
- **docs/** - Technical documentation
- **CHANGELOG.md** - List your changes

### Documentation Style

```python
def generate_personalized_lab(
    student_id: str,
    topic_name: str,
    concepts: List[str]
) -> Dict[str, Any]:
    """
    Generate a personalized lab based on student mastery and hobbies.
    
    This function integrates outputs from SQKT, G-CDM, and RL Agent to create
    a customized lab experience tailored to the student's current knowledge
    state and personal interests.
    
    Args:
        student_id: Unique identifier for the student
        topic_name: Name of the topic (e.g., "NoSQL Databases")
        concepts: List of concept names to include in the lab
        
    Returns:
        Dictionary containing:
            - lab_id: Unique lab identifier
            - title: Lab title
            - sections: List of lab sections with exercises
            - estimated_time: Total time in minutes
            - difficulty: Overall difficulty level
            
    Raises:
        ValueError: If student_id is empty or concepts list is empty
        ServiceUnavailableError: If AI services are not responding
        
    Example:
        >>> lab = generate_personalized_lab(
        ...     student_id='student_123',
        ...     topic_name='NoSQL Databases',
        ...     concepts=['CAP Theorem', 'Data Modeling']
        ... )
        >>> print(lab['title'])
        'Personalized Lab: NoSQL Databases'
    """
    # Implementation
    pass
```

---

## 🐛 Reporting Bugs

### Before Reporting

1. Check if the bug has already been reported
2. Try to reproduce the bug
3. Gather relevant information

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. Go to '...'
2. Click on '...'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**
- OS: [e.g., macOS 14.0]
- Python version: [e.g., 3.12]
- Browser: [e.g., Chrome 120]

**Additional context**
Any other relevant information.
```

---

## 💡 Feature Requests

### Feature Request Template

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
What you want to happen.

**Describe alternatives you've considered**
Other solutions you've thought about.

**Additional context**
Any other relevant information.
```

---

## 🎯 Development Workflow

### Typical Workflow

1. **Pick an issue** - Choose from "good first issue" or "help wanted"
2. **Discuss** - Comment on the issue to claim it
3. **Branch** - Create a feature branch
4. **Develop** - Write code and tests
5. **Test** - Ensure all tests pass
6. **Document** - Update relevant documentation
7. **Commit** - Make clear, atomic commits
8. **Push** - Push to your fork
9. **PR** - Open a pull request
10. **Review** - Address feedback
11. **Merge** - Maintainers will merge when ready

---

## 📞 Getting Help

- **GitHub Issues** - For bugs and feature requests
- **Discussions** - For questions and ideas
- **Documentation** - Check docs/ folder

---

## 🙏 Thank You!

Your contributions make KTCD_Aug better for everyone. We appreciate your time and effort!

---

**Happy Coding! 🚀**

