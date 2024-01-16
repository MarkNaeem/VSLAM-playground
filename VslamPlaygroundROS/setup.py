from setuptools import find_packages, setup

package_name = 'VslamPlaygroundROS'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='marknaeem97',
    maintainer_email='marknaeem@yahoo.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'data_publisher_py = VslamPlaygroundROS.data_publisher_py:main'
        ],
    },
)
