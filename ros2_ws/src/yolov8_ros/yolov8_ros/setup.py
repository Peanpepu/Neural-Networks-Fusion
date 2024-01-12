from setuptools import setup

package_name = 'yolov8_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Pedro Antonio Peña Puente',
    maintainer_email='pedrop@uma.es',
    description='YOLOv8 for ROS 2',
    license='GPL-3',
    entry_points={
        'console_scripts': [
                # 'prueba_ros = yolov8_ros.prueba_ros:main',
                'fusion_ros = yolov8_ros.fusion_ros:main',
                'img_publisher = yolov8_ros.img_publisher:main',
                'visualization = yolov8_ros.visualization:main',
                'save_img = yolov8_ros.save_img:main'
        ],
    },
)
