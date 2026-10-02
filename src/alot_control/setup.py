from setuptools import find_packages, setup

package_name = 'alot_control'

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
    maintainer='seungwon',
    maintainer_email='jws10375@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'alot_arm_control_node = alot_control.alot_arm_control_node:main',
        'alot_motor_control_node = alot_control.alot_motor_control_node:main',
        'box_detect_node = alot_control.box_detect_node:main',
        'main2 = alot_control.main2:main',
        ],
    },
)
