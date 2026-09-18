from setuptools import find_packages, setup

package_name = 'aiot_heuristic_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/boxes_sample_m.json']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='roma',
    maintainer_email='badukjoo@naver.com',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'heuristic_main_node = aiot_heuristic_pkg.heuristic_main_node:main',
            'heuristic_debug_node = aiot_heuristic_pkg.heuristic_debug_node:main',
        ],
    },
)
