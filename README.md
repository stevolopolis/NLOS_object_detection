# shadow_cam

Moving obstacles occluded by corners are a potential source for collisions in mobile robotics applications such as autonomous vehicles. In this paper, we address the problem of anticipating such collisions by proposing a vision-based detection algorithm for obstacles which are outside of a vehicle's direct line of sight. Our method detects shadows of obstacles hidden around corners and automatically classifies these unseen obstacles as ``dynamic`` or ``static``. We evaluate our proposed detection algorithm on real-world corners and a large variety of simulated environments to assess generalizability in different challenging surface and lighting conditions. The mean classification accuracy on simulated data is around 80% and on real-world corners approximately 70%. Additionally, we integrate our detection system on a full-scale autonomous wheelchair and demonstrate its feasibility as an additional safety-mechanism through real-world experiments. We are open-sourcing our real-time capable implementation of the proposed ShadowCam algorithm and the dataset containing simulated and real-world data.

![teaser](teaser-itsc-2018.jpg)

# Installation

 - Install ROS
 - Download [usb_cam](https://github.com/ros-drivers/usb_cam/tree/develop)
 - Install [AprilTags](apriltag-2016-12-01) (included from [AprilTag2](https://april.eecs.umich.edu/software/apriltag.html))
 - ``catkin_make``

