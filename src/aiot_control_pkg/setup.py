from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'aiot_control_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),

        # launch 파일 설치
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')
        ),

        # config yaml 파일 설치
        (
            os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')
        ),

        # urdf 파일 설치
        (
            os.path.join('share', package_name, 'urdf'),
            glob('urdf/*.urdf')
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pc',
    maintainer_email='jisu034@naver.com',
    description='AIOT 5DOF robot arm current control package using DH parameters and Dynamixel X-series motors',
    license='TODO',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # 'current_control_node = aiot_control_pkg.current_control_node:main',
            # 'current_motor_interface_node = aiot_control_pkg.current_motor_interface_node:main',
            # 'current_ikpy_4dof_node = aiot_control_pkg.current_ikpy_4dof_node:main',
            # 'a_position_ikpy_4dof_node = aiot_control_pkg.a_position_ikpy_4dof_node:main',
            'motor_control_node = aiot_control_pkg.motor_control_node:main',
            'motor_compliance_node = aiot_control_pkg.motor_compliance_node:main',
            'aiot_compliance_node = aiot_control_pkg.aiot_compliance_node:main',
            
        ],
    },
)
