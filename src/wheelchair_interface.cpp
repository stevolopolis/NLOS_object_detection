#include <ros/ros.h>
#include <std_msgs/Bool.h>


std::string default_node_name = "wheelchair_interface_node";
std::string pub_topic_user_input = "user_input_autonomous_driving_on";
std::string sub_topic_shadow_cam = "shadow_cam_movement_detection";


bool auto_mode_on = true;
ros::Time last_cmd_received;
double watchdog_timeout = 3;

void shadowCamCallback(const std_msgs::Bool::ConstPtr &detection) {
  auto_mode_on = !detection->data;
  last_cmd_received = ros::Time::now();
}


int main(int argc, char **argv) {
  ros::init(argc, argv, default_node_name);
  ros::NodeHandle n;
  ros::Rate loop_rate(50);

  ros::Subscriber sub = n.subscribe(sub_topic_shadow_cam, 1, &shadowCamCallback);
  ros::Publisher pub = n.advertise<std_msgs::Bool>(pub_topic_user_input, 1);

  std_msgs::Bool msg;

  while(ros::ok()) {

    //publish stop if movement detected
    msg.data = auto_mode_on;
    if (ros::Time::now().toSec() - last_cmd_received.toSec() > watchdog_timeout) {
      msg.data = true;
    }
    pub.publish(msg);
    ROS_INFO("detection [%d]", msg.data);

    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}

