"""
Data Poisoning Toolkit - A Comprehensive Framework for ML Security Research
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="datapoisontoolkit",
    version="1.0.0",
    author="Muhammet Anil Yagiz, Muhammed Arif Dayi, Dr. Polat Goktas",
    author_email="anill.yagiz@gmail.com",  
    description="A Comprehensive Framework for ML Security Research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/[username]/data-poison-toolkit",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "research": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "notebook>=6.4.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.900",
        ],
    },
    entry_points={
        "console_scripts": [
            "datapoisontoolkit=datapoisontoolkit.demo_runner:main",
        ],
    },
    keywords=[
        "machine learning",
        "security",
        "data poisoning",
        "adversarial attacks",
        "defense mechanisms",
        "research",
        "pytorch",
        "scikit-learn",
    ],
    project_urls={
        "Bug Reports": "https://github.com/arifs122/data-poison-toolkit/issues",
        "Source": "https://github.com/arifs122/data-poison-toolkit",
        "Documentation": "https://github.com/arifs122/data-poison-toolkit#readme",
        "Research Paper": "https://github.com/arifs122/data-poison-toolkit/blob/main/paper.md",
    },
    include_package_data=True,
    package_data={
        "datapoisontoolkit": [
            "data/*.csv",
            "data/*.json",
            "config/*.yaml",
        ],
    },
)