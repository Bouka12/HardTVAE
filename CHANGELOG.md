# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-05

### Added
- Initial release of HardVAE framework
- Core hardness calculation module with 18+ PyHard metrics
- Custom hardness metrics: relative entropy and PCA-based contributions
- Comprehensive evaluation framework with 6 evaluation dimensions:
  - Statistical fidelity (distribution matching)
  - Topological fidelity (shape preservation via persistent homology)
  - Instance-level fidelity (hardness pattern similarity)
  - Complexity fidelity (data complexity metric preservation)
  - Utility evaluation (downstream task performance)
  - Clustering fidelity (cluster structure preservation)
- Hardness-aware CVAE architecture for tabular data synthesis
- Multiple weighting strategies: static, curriculum, self-paced
- Comprehensive visualization and reporting capabilities
- Fidelity analysis scripts for distributional, hardness, and topological metrics
- Complete documentation and examples
- Unit test suite
- Development guidelines and contributing instructions

### Features
- **HardnessCalculator**: Flexible hardness metric computation
- **SyntheticDataEvaluator**: Multi-dimensional quality assessment
- **TabularCVAE**: Conditional VAE for tabular data
- **HardnessAwareCVAETrainer**: Hardness-weighted training
- **Visualization tools**: Dashboards, heatmaps, radar charts
- **Analysis utilities**: Distributional, hardness, and topological analysis

### Documentation
- Comprehensive README with quick start guide
- API reference documentation
- Detailed methodology explanation
- Project structure guide
- Contributing guidelines
- Example scripts and tutorials


---

## How to Report Changes

When contributing changes, please:

1. Update this file with your changes
2. Follow the format: `### [Type]` where Type is one of:
   - `Added`: New features
   - `Changed`: Changes in existing functionality
   - `Deprecated`: Soon-to-be removed features
   - `Removed`: Removed features
   - `Fixed`: Bug fixes
   - `Security`: Security fixes

3. Include version number and date
4. Reference related issues with `#issue_number`

## Version History

- **1.0.0** (2025-12-05): Initial release
