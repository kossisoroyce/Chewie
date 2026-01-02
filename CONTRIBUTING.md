# Contributing to AfriMed CHW Assistant

Thank you for your interest in contributing to AfriMed! This project aims to improve maternal healthcare in Africa through AI-powered support for Community Health Workers.

## How to Contribute

### Reporting Issues
- Use GitHub Issues to report bugs or suggest features
- Include as much detail as possible
- For medical accuracy concerns, please tag with `medical-review`

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**
4. **Run tests**: `pytest tests/`
5. **Submit a pull request**

### Areas We Need Help With

#### High Priority
- **Medical Content Review**: Native Swahili speakers with medical background to validate translations
- **Training Data**: CHWs willing to share anonymized interaction logs
- **Language Expansion**: Hausa, Yoruba, Amharic speakers for new language support

#### Code
- Improving danger sign detection accuracy
- Adding new communication channels (USSD, voice)
- Performance optimization for low-bandwidth environments
- Unit tests and integration tests

#### Documentation
- User guides for CHWs
- API documentation
- Translation of documentation to local languages

### Medical Content Guidelines

⚠️ **Important**: All medical content must be:
- Based on WHO or national health ministry guidelines
- Reviewed by a qualified healthcare professional
- Never provide diagnostic conclusions
- Always recommend facility referral for danger signs

### Code Style

- Follow PEP 8 for Python code
- Use type hints
- Write docstrings for all public functions
- Keep functions focused and testable

### Commit Messages

Use conventional commits:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring

Example: `feat: add Hausa language support for danger signs`

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Remember the end users are CHWs serving vulnerable populations
- Prioritize safety in all medical-related decisions

## Getting Help

- Open an issue for questions
- Tag maintainers for urgent medical accuracy concerns
- Join our community discussions

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
