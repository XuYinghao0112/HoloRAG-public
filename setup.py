import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="holorag",
    version="0.1.0",
    author="xyh",
    description="HoloRAG: hierarchical heterogeneous graph RAG with granularity-aware adaptive reasoning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://local/holorag",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    python_requires=">=3.10",
    install_requires=[
        "openai>=1.91.0,<1.92.0",
        "torch==2.5.1",
        "transformers==4.45.2",
        "networkx==3.4.2",
        "httpx>=0.27.0,<0.29.0",
        "numpy>=1.24.0,<3.0.0",
        "tqdm>=4.66.0",
        "accelerate>=0.33.0",
        "safetensors>=0.4.3",
    ]
)
