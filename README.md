<p align="center">
<img src="assets/deepcompressor.png" alt="DeepCompressor Logo" width="450">
</p>

<h2><p align="center">Model Compression Toolbox for LLM, Diffusion Models and VAR Models</p></h2>

<p align="center">
    <a href="https://github.com/mit-han-lab/deepcompressor/blob/master/LICENSE">
        <img alt="Apache License" src="https://img.shields.io/github/license/mit-han-lab/deepcompressor">
    </a>
    <!-- <a href="https://deepcompressor.mit.edu">
        <img alt="Website" src="https://img.shields.io/website?up_message=deepcompressor&url=https%3A%2F%2Fdeepcompressor.mit.edu">
    </a> -->
   <!-- <a href="https://pypi.org/project/deepcompressor/">
        <img alt="Pypi" src="https://img.shields.io/pypi/v/deepcompressor">
    </a> -->
</p>

# About This Fork
This repository is a specialized fork of the DeepCompressor framework, tailored specifically to democratize high-fidelity Visual Autoregressive (VAR) models for edge deployment.

* *What this fork adds:* We introduce a comprehensive W4A4 and INT8 KV-cache quantization pipeline specifically designed for the Infinity family of generative models (2B and 8B). It mitigates extreme activation outliers using SVDQuant and compresses the monotonically growing KV-cache via Asymmetric Per-Channel INT8 Quantization. This allows the 8B model to run natively on 16GB edge silicon.
* *Paper:* For full methodological details, evaluation metrics, and edge hardware deployment strategies on NVIDIA Jetson architectures, please refer to our paper: Enabling 8B Bitwise Autoregressive Image Generation on Edge GPUs.

