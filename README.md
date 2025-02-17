# FIGURE: Form, Invariance, and Geometric Understanding for Representation Evaluation

FIGURE is a synthetic dataset designed for studying **shape-based and motion-based representation learning** while controlling for **texture bias and global transformations**. Each sample in FIGURE is modeled as a **hierarchical kinematic structure**, where limbs are defined relative to a central **torso frame**. The dataset allows for controlled evaluations of robustness to **texture bias, global transformations, and motion dynamics**.  

Each sample is defined by a **set of relative limb frames**, describing the spatial relationships between body parts. In some variants, figures also undergo **global transformations**, where the entire body is placed within a world coordinate frame using SE(2) transformations. This enables experiments on **pose-invariant learning** while distinguishing between **relative shape structure and absolute world position**.  

FIGURE consists of two main tasks:  

1. **Shape Classification:** Figures are classified based on their **relative limb configurations** (e.g., arms up vs. arms down).  
2. **Motion Classification:** Figures are classified based on **sequences of shape frames evolving over time**, representing distinct motion patterns.  

To evaluate different aspects of representation learning, FIGURE includes the following sub-datasets:  

## Shape Classification  

**FIGURE-Shape-B**:  ![FIGURE-Shape-B](datasets/visualizations/FIGURE-Shape-B.gif) ![FIGURE-Shape-B](datasets/visualizations/FIGURE-Shape-B-test.gif) 

The <ins>b</ins>ase **shape classification** dataset. Figures are classified based on their **shape**, which is defined by the arrangement of **limb frames** (e.g., arms up vs. arms down). There are no global transformations or color biases.

**FIGURE-Shape-CB**:  ![FIGURE-Shape-B](datasets/visualizations/FIGURE-Shape-CB.gif) ![FIGURE-Shape-B](datasets/visualizations/FIGURE-Shape-CB-test.gif) ![FIGURE-Shape-B](datasets/visualizations/FIGURE-Shape-CB-test-bias.gif) 

A variant introducing a **<ins>c</ins>olor <ins>b</ins>ias** in training, where a specific class (e.g., arms-up figures) is more often associated with a particular shirt color. At test time, this correlation is inverted, allowing for an evaluation of **texture-invariant learning**.  

**FIGURE-Shape-PI**:  ![FIGURE-Shape-B](datasets/visualizations/FIGURE-Shape-PI.gif) ![FIGURE-Shape-B](datasets/visualizations/FIGURE-Shape-PI-test.gif) 

A shape classification dataset with **global transformations**, where figures undergo **random translations and rotations** in SE(2). The relative limb frames remain unchanged, but the torso frame is placed randomly in the world frame. This tests **<ins>p</ins>ose-<ins>i</ins>nvariant learning**.  

**FIGURE-Shape-F**: ![FIGURE-Shape-B](datasets/visualizations/FIGURE-Shape-F.gif) ![FIGURE-Shape-B](datasets/visualizations/FIGURE-Shape-F-test.gif) ![FIGURE-Shape-B](datasets/visualizations/FIGURE-Shape-F-test-bias.gif) 

The <ins>f</ins>ull set of variations by combination of **both color bias and global transformations**. This variant tests the ability to disentangle **shape, texture, and absolute position** when learning representations.  

FIGURE provides a **controlled and flexible testbed** for studying **equivariance, invariance, and generalization** in deep learning models. Future extensions may introduce additional **motion complexity, 3D representations, and real-world domain adaptation**.  

## License  
This dataset is licensed under the **[CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/)**.  

## Citation  
If you use FIGURE in your research, please cite it as:  

```
@misc{figure2025, title={FIGURE: Form, Invariance, and Geometric Understanding for Representation Evaluation}, author={Erik J. Bekkers}, year={2025}, howpublished={\url{https://github.com/ebekkers/FIGURE}}, }
```
