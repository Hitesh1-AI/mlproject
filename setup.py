from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str)->List[str]:
    '''
    this func will returm the list of requirements
    '''
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        [req.replace('\n', "") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

setup(
    name= "mlproject",
    version= '0.0.1',
    author= "Hitesh",
    author_email= "hiteshkumar78785@gmail.com",
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt')

)