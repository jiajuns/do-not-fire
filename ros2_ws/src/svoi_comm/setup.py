from setuptools import setup
import os
from glob import glob

package_name = 'svoi_comm'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.sh')),
        (os.path.join('share', package_name, 'integration'),
         glob('integration/*.patch') + glob('integration/*.md')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='hq',
    maintainer_email='hq10606@uga.edu',
    description='SVoI communication scheduling for bandwidth-limited multi-robot exploration',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'comm_node = svoi_comm.comm_node:main',
            'map_gate_node = svoi_comm.map_gate_node:main',
            'pose_file_writer = svoi_comm.pose_file_writer:main',
        ],
    },
)
