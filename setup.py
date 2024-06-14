from setuptools import find_packages, setup


def get_install_requirements():
    return [
        "einops",
        "numpy",
        "pyyaml",
        "torch",
        "torchvision",
        "scipy",
        "tqdm",
        "wandb",
        "threed-front",
    ]


def setup_package():
    with open("README.md") as f:
        long_description = f.read()
    setup(
        name='MiDiffusion',
        maintainer="Siyi Hu",
        maintainer_email="siyihu02@gmail.com",
        version='0.1',
        license='BSD-2-Clause',
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering",
            "Programming Language :: Python :: 3.8",
        ],
        description='Mixed diffusion models for synthetic 3D scene generation',
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=find_packages(include=['midiffusion']),
        install_requires=get_install_requirements(),
    )


if __name__ == "__main__":
    setup_package()
