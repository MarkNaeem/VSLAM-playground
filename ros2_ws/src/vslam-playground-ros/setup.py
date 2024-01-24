from setuptools import find_packages, setup
import os
import glob
package_name = 'vslam-playground-ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name), glob('config/*config.rviz')),          
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='marknaeem',
    maintainer_email='marknaeem@yahoo.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'data_publisher_py = src.data_publisher_py:main'
        ],
    },
)
