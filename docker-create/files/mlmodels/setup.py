from setuptools import setup, find_packages
from pathlib import Path

# 读取 requirements.txt（忽略空行和注释）
def parse_requirements(filename):
    path = Path(__file__).parent / filename
    requirements = []
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append(line)
    return requirements

setup(
    name='yolov5_local',
    version='7.0',
    description='YOLOv5 Custom Local Version',
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    python_requires='>=3.8',
)