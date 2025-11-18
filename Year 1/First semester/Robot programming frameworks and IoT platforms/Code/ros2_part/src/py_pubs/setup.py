import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'py_pubs'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
         # Include all launchfiles.
        (os.path.join('share', package_name, 'launch'),
        glob(os.path.join('launch', '*launch.[pxy][yma]*'))
        ) # assumes_launchsuffix
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='brairlab',
    maintainer_email='brairlab@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "talker = py_pubs.publisher_member_function:main",
            "listener = py_pubs.subscriber_member_function:main"
        ],
    },
)
