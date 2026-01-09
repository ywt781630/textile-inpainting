# Textile Inpainting

This repository contains the implementation of our proposed method for digital restoration of ancient textile images with large missing regions and complex structural patterns.

---

## Overview

The restoration of ancient textile images is challenging due to severe degradation, extensive missing areas, and intricate structural motifs. Existing image inpainting methods often struggle to preserve global structural coherence while maintaining fine-grained texture details. To address these challenges, we propose a structure-guided restoration framework specifically designed for textile cultural heritage images.

---

## Method Overview

Our method adopts a **three-stage restoration architecture** that progressively reconstructs textile images by decoupling structure modeling and texture synthesis:

1. **Structure Generation Stage**  
   In the first stage, the global structural layout of the textile image is modeled. We introduce a codebook-based vector quantization strategy to discretize intermediate structural representations, which enables efficient Transformer-based modeling while reducing computational complexity. This stage focuses on capturing long-range dependencies and preserving the semantic consistency of repetitive textile patterns.

2. **Texture Refinement Stage**  
   Given the generated structural guidance, the second stage emphasizes local texture synthesis. Convolutional neural networks are employed to recover fine-grained textile details, ensuring that the restored regions are visually coherent with surrounding textures and material characteristics.

3. **Image Completion Stage**  
   In the final stage, structure and texture features are fused to produce the completed image. This stage further refines boundary transitions and improves overall visual fidelity, resulting in restorations that are both structurally consistent and perceptually realistic.

Through this staged design, the proposed framework effectively balances global structure preservation and local texture reconstruction, making it particularly suitable for complex textile restoration scenarios.

---

## Code

This repository currently includes:
- Partial implementation of the proposed network architecture  

Additional components, including full training code, complete dataset descriptions, and extended evaluation scripts, will be **continuously updated** in future revisions of this repository.

---

## Dataset

The dataset used in this work consists of ancient textile images acquired through a hybrid image acquisition protocol that combines field photography and laboratory imaging. Detailed dataset specifications and download instructions will be provided in subsequent updates.

---

## Citation

If you find this work useful for your research, please consider citing our paper. Citation information will be added upon publication.

---

## License

This project is released for academic research purposes only. License details will be specified in a future update.
