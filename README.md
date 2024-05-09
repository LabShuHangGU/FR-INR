# FR-INR
This repository is the official PyTorch implementation of "Improved Implicit Neural Representation with Fourier Reparameterized Training", CVPR, 2024.
By Kexuan Shi, Xingyu Zhou, Shuhang Gu.
([arxiv](https://arxiv.org/pdf/2401.07402.pdf))


**Abstract:** Implicit Neural Representation (INR) as a mighty representation paradigm has achieved success in various computer vision tasks recently. Due to the low-frequency bias issue of vanilla multi-layer perceptron (MLP), existing methods have investigated advanced techniques, such as positional encoding and periodic activation function, to improve the accuracy of INR. In this paper, we connect the network training bias with the reparameterization technique and theoretically prove that weight reparameterization could provide us a chance to alleviate the spectral bias of MLP. Based on our theoretical analysis, we propose a **Fourier reparameterization** method which learns coefficient matrix of fixed Fourier bases to compose the weights of MLP. We evaluate the proposed Fourier reparameterization method on different INR tasks with various MLP architectures, including vanilla MLP, MLP with positional encoding and MLP with advanced activation function, etc. The superiority approximation results on different MLP architectures clearly validate the advantage of our proposed method. Armed with our Fourier reparameterization method, better INR with more textures and less artifacts can be learned from the training data. 

## Codes will be coming soon!




## Citation
    @misc{shi2024improved,
      title={Improved Implicit Neural Representation with Fourier Reparameterized Training}, 
      author={Kexuan Shi and Xingyu Zhou and Shuhang Gu},
      year={2024},
      eprint={2401.07402},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
