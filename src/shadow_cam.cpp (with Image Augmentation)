
#include "apriltag.h"
#include "tag36h11.h"
#include "tag36h10.h"
#include "tag36artoolkit.h"
#include "tag25h9.h"
#include "tag25h7.h"
#include "tag16h5.h"

#include "common/getopt.h"
#include "common/image_u8.h"
#include "common/image_u8x4.h"
#include "common/pjpeg.h"
#include "common/zarray.h"

#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <opencv2/calib3d/calib3d.hpp>

using namespace cv;

std::string node_name = "shadow_cam_node";
std::string sub_image_topic = "sub_image_topic";
std::string pub_topic_shadow_cam = "shadow_cam_movement_detection";

apriltag_detector_t *td = apriltag_detector_create();
apriltag_family_t *tf = tag16h5_create();

std::vector<double> check_seq_sums;

ros::Publisher pub;

struct params {
  int max_tags = 6;
  int corner_tag_id = 4;
  int ignore_tag_id = -1;
  double threshold = 15.;
  double magnification = 10.;
  int dilate = 1;
  int erode = 3;
  int seq_length = 10;
  double tag_size = 21.0;
  double cropping_width = 175.0;
  double cropping_height = 190.0;
  double cropping_delta_h = -70.0;
  double cropping_delta_w = 15.0;
  int seq_idx = 0;
  int seq_counter = 0;
  int tag_counter = 0;
  int lost_frames_counter = 0;
  bool debug_on_max = false;
  bool debug_on_min = false;
  bool debug_on_sum = true;
  std::vector<cv::Mat> tmp_seq_imgs;
  std::vector<std::map<int, std::vector<cv::Point2f> > > tmp_seq_dets;
} p;


cv::Mat clip_img(cv::Mat img) {
  cv::Mat img_out;
  img.convertTo(img_out, CV_8UC3);
  img_out.setTo(0, img < 0);
  img_out.setTo(255, img > 255);
  return img_out;
}


void get_scaling(int &h_px, int &w_px, int &dh_px, int &dw_px) {
  cv::Point2f a(p.tmp_seq_dets[0][p.corner_tag_id][0].x, p.tmp_seq_dets[0][p.corner_tag_id][0].y);
  cv::Point2f b(p.tmp_seq_dets[0][p.corner_tag_id][1].x, p.tmp_seq_dets[0][p.corner_tag_id][1].y);
  cv::Point2f c(p.tmp_seq_dets[0][p.corner_tag_id][2].x, p.tmp_seq_dets[0][p.corner_tag_id][2].y);

  double tmp_h_px = cv::norm(a-b);
  double tmp_w_px = cv::norm(b-c);

  if (p.debug_on_max) {
    std::cout << tmp_h_px << std::endl;
    std::cout << tmp_w_px << std::endl;
  }

  h_px  = int(p.cropping_height   / p.tag_size * tmp_h_px);
  w_px  = int(p.cropping_width    / p.tag_size * tmp_w_px);

  dh_px = int(p.cropping_delta_h  / p.tag_size * tmp_h_px);
  dw_px = int(p.cropping_delta_w  / p.tag_size * tmp_w_px);
}


cv::Mat get_cropped_img(cv::Mat img, std::vector<cv::Point2f> corners) {
  int h_px, w_px, dh_px, dw_px;
  get_scaling(h_px, w_px, dh_px, dw_px);

  int startX = round(corners[0].x) + dw_px;
  int startY = round(corners[0].y) + dh_px;
  cv::Mat result;

  try {
    result = img(cv::Rect(startX,
                          startY,
                          w_px,
                          h_px));
  } catch (cv::Exception& e) {
    ROS_ERROR("Cropping failed.");
  }

  if (p.debug_on_max) {
    cv::imshow("before cropping", img);
    cv::imshow("after cropping", result);
    cv::waitKey(100);
  }

  return result;
}


cv::Mat get_mean_img(std::vector<cv::Mat> seq) {

  cv::Mat mean_img = seq[0];
  mean_img.convertTo(mean_img, CV_64FC3);

  for (int i = 1; i < seq.size(); i++) {
    cv::Mat tmp_img;
    seq[i].convertTo(tmp_img, CV_64FC3);
    mean_img += tmp_img;
  }

  mean_img /= double(seq.size());

  return mean_img;
}


cv::Mat getElement(int size) {
  cv::Mat element = cv::getStructuringElement(MORPH_ELLIPSE,
                                              cv::Size(2*size+1, 2*size+1),
                                              cv::Point(size, size));
  return element;
}


void print_vec(std::vector<double> vec) {
  std::cout << '[';
  for (std::vector<double>::const_iterator i = vec.begin(); i != vec.end(); ++i) {
    std::cout << *i << ',';
  }
  std::cout << ']' << std::endl;
}


cv::Mat resize(cv::Mat img) {
  cv::Mat result;
  cv::resize(img, result, cv::Size(100,100), 0, 0, CV_INTER_LINEAR); //TODO param
  return result;
}


//https://stackoverflow.com/questions/23510571/how-to-set-given-channel-of-a-cvmat-to-a-given-value-efficiently-without-chang
void setChannel(cv::Mat &mat) {

  cv::Scalar m, d;
  cv::meanStdDev(mat, m, d);

  for (int y=0; y < mat.rows; y++) {
    for (int x=0; x < mat.cols; x++) {
      for (int c=0; c < mat.channels(); c++) {
        if ( abs(mat.at<cv::Vec3d>(y,x)[c] - float(m.val[c])) > 2*float(d.val[c])) {
          mat.at<cv::Vec3d>(y,x)[c] = 255;
        } else {
          mat.at<cv::Vec3d>(y,x)[c] = 0;
        }
      }
    }
  }

  return;
}


bool color_amp_detection(std::vector<cv::Mat> seq) {

  cv::Mat mean_img = get_mean_img(seq);
  double seq_sum = 0.;
  bool result = false;

  for (int i = 0; i < seq.size(); i++) {
    cv::Mat tmp_img;
    cv::Mat tmp_sub;
    seq[i].convertTo(tmp_img, CV_64FC3);
    seq[i].convertTo(tmp_sub, CV_64FC3);

    // color amp
    cv::GaussianBlur( (tmp_img - mean_img), tmp_sub, Size(3,3), 0, 0); //TODO param
    tmp_sub *= p.magnification;
    tmp_sub = cv::abs(tmp_sub);

    setChannel(tmp_sub);

    int dilate_size = p.dilate;
    int erode_size = p.erode;
    cv::dilate(tmp_sub, tmp_sub, getElement(dilate_size));
    cv::erode(tmp_sub, tmp_sub, getElement(erode_size));

    seq_sum += cv::sum(cv::sum(tmp_sub))[0];

    if (p.debug_on_max || p.debug_on_min) {
      cv::imshow("mean", clip_img(mean_img));
      cv::imshow("seq", clip_img(seq[i]));
      cv::imshow("tmp_sub", clip_img(tmp_sub));
      cv::waitKey(100);
    }
  }

  if (p.debug_on_max || p.debug_on_sum) {
    check_seq_sums.push_back(seq_sum);
    print_vec(check_seq_sums);
  }

  if (seq_sum > p.threshold) {
    result = true;
  }

  return result;
}


cv::Mat get_h(std::map<int, std::vector<cv::Point2f> > first_detection_map,
              std::map<int, std::vector<cv::Point2f> > good_detection_map) {

  std::vector<cv::Point2f> obj;
  std::vector<cv::Point2f> scene;
  cv::Mat h;

  std::map<int, std::vector<cv::Point2f> >::iterator it;

  for( it = good_detection_map.begin(); it != good_detection_map.end(); it++ ) {

    if ( first_detection_map.find(it->first) != first_detection_map.end() ) {

      for (int i = 0; i < first_detection_map[it->first].size(); i++) {
        obj.push_back(it->second[i]);
        scene.push_back(first_detection_map[it->first][i]);
      }
    }
  }

  h = cv::findHomography( obj, scene, CV_RANSAC ); //TODO param

  return h;
}


std::vector<cv::Mat> build_tmp_seq(std::vector<cv::Mat> tmp_seq_imgs,
                                   std::vector<std::map<int, std::vector<cv::Point2f> > > tmp_seq_dets) {
  std::vector<cv::Mat> result;
  std::vector<std::map<int, std::vector<cv::Point2f> > >::size_type d = 0;
  cv::Mat tmp_img;
  cv::Mat img_out_cropped;
  cv::Size tmp_s = tmp_seq_imgs.front().size();

  for(std::vector<cv::Mat>::size_type i = 0; i != tmp_seq_imgs.size(); i++) {
    if (i == 0) {
      tmp_img = tmp_seq_imgs[i];
      img_out_cropped = get_cropped_img(tmp_img, tmp_seq_dets[0][p.corner_tag_id]);

    } else {
      cv::Mat h = get_h(tmp_seq_dets[0], tmp_seq_dets[d]);
      cv::warpPerspective(tmp_seq_imgs[i], tmp_img, h, tmp_s);
      img_out_cropped = get_cropped_img(tmp_img, tmp_seq_dets[0][p.corner_tag_id]);

      if (p.debug_on_max) {
        cv::imshow("after h", clip_img(tmp_img));
        cv::waitKey(100);
      }
    }

    if (!img_out_cropped.empty()) {
      result.push_back(resize(img_out_cropped.clone()));
    } else {
      p.lost_frames_counter += 1;
    }

    d++;
  }

  return result;
}


void apriltag_seq(cv::Mat& image) {

  p.tag_counter = 0;
  bool pre_req = false;
  std::map<int, std::vector<cv::Point2f> > good_detection_map;
  ros::Time time_start_seq;
  ros::Time time_end_seq;
  ros::Time t_start = ros::Time::now();

  cv::Mat gray(image.size(), CV_MAKETYPE(image.depth(), 1));
  cv::cvtColor(image, gray, CV_BGR2GRAY);

  image_u8_t im = { .width = gray.cols,
                    .height = gray.rows,
                    .stride = gray.cols,
                    .buf = gray.data
                  };

  zarray_t *detections = apriltag_detector_detect(td, &im);

  for (int i = 0; i < zarray_size(detections); i++) {
    apriltag_detection_t *det;
    zarray_get(detections, i, &det);

    if (det->hamming == 0 && det->id != p.ignore_tag_id) {

      std::vector<cv::Point2f> imgPts;
      std::pair<float, float> p1 = std::make_pair(det->p[0][0], det->p[0][1]);
      std::pair<float, float> p2 = std::make_pair(det->p[1][0], det->p[1][1]);
      std::pair<float, float> p3 = std::make_pair(det->p[2][0], det->p[2][1]);
      std::pair<float, float> p4 = std::make_pair(det->p[3][0], det->p[3][1]);
      imgPts.push_back(cv::Point2f(p1.first, p1.second));
      imgPts.push_back(cv::Point2f(p2.first, p2.second));
      imgPts.push_back(cv::Point2f(p3.first, p3.second));
      imgPts.push_back(cv::Point2f(p4.first, p4.second));

      good_detection_map.insert(std::make_pair(det->id, imgPts));

      p.tag_counter += 1;

      if (p.tag_counter >= p.max_tags && good_detection_map.find(p.corner_tag_id) != good_detection_map.end()) {
        pre_req = true;
      }
    }

    if (p.debug_on_max) {
      printf("\ndetection %3d: id (%2dx%2d)-%-4d, hamming %d, goodness %8.3f, margin %8.3f\n",
             i, det->family->d*det->family->d,
             det->family->h, det->id, det->hamming,
             det->goodness, det->decision_margin);
    }
  }

  apriltag_detections_destroy(detections);

  if (p.debug_on_max) {
    cv::imshow("apriltag image", image);
    cv::waitKey(100);
  }

  if (pre_req) {

    if (p.seq_idx < p.seq_length) {
      p.tmp_seq_imgs.push_back(image.clone());
      p.tmp_seq_dets.push_back(good_detection_map);

      p.seq_idx += 1;

    } else if (p.seq_idx == p.seq_length) {
      time_start_seq = ros::Time::now();

      std::vector<cv::Mat> tmp_seq = build_tmp_seq(p.tmp_seq_imgs, p.tmp_seq_dets);
      
      std_msgs::Bool msg;
      if (color_amp_detection(tmp_seq)) {
        ROS_WARN("ShadowCam: MOVEMENT DETECTED");
        //TODO https://docs.opencv.org/2.4/doc/tutorials/introduction/display_image/display_image.html
        msg.data = true;
      } else {
        msg.data = false;
      }
      pub.publish(msg);

      p.seq_counter += 1;

      time_end_seq = ros::Time::now();
      ros::Duration d_seq = time_end_seq - time_start_seq;
      ROS_INFO("seq [time: %f]", d_seq.toSec());

      p.tmp_seq_imgs.erase(p.tmp_seq_imgs.begin());
      p.tmp_seq_dets.erase(p.tmp_seq_dets.begin());

      p.tmp_seq_imgs.push_back(image.clone());
      p.tmp_seq_dets.push_back(good_detection_map);

    } else {
      ROS_ERROR("[seq_idx %d] out of range", p.seq_idx);
    }

  } else {
    if (p.debug_on_max) {
      ROS_WARN("Frame doesn't fulfill prerequisites: Dropped.");
    }
    p.lost_frames_counter += 1;
  }

  ros::Time t_end = ros::Time::now();
  ros::Duration d_img = t_end - t_start;

  if (p.debug_on_max) {
    ROS_INFO("seq_idx [%d] seq_counter [%d] lost_frames_counter [%d] seq_length [%d] time [%f]", p.seq_idx, p.seq_counter, p.lost_frames_counter, int(p.tmp_seq_imgs.size()), d_img.toSec());
  }
}


void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
  try {
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    cv::Mat frame = cv_ptr->image;

    apriltag_seq(frame);

  } catch (cv_bridge::Exception & e) {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}


int main(int argc, char * * argv) {

  ros::init(argc, argv, node_name);
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  private_nh.getParam("max_tags", p.max_tags);
  private_nh.getParam("corner_tag_id", p.corner_tag_id);
  private_nh.getParam("threshold", p.threshold);
  private_nh.getParam("magnification", p.magnification);
  private_nh.getParam("ignore_tag_id", p.ignore_tag_id);
  private_nh.getParam("debug_on_max", p.debug_on_max);
  private_nh.getParam("debug_on_min", p.debug_on_min);
  private_nh.getParam("debug_on_sum", p.debug_on_sum);

  apriltag_detector_add_family(td, tf);
  td->quad_decimate = 1.0;
  td->quad_sigma = 0.0;
  td->refine_edges = 1;
  td->refine_decode = 0;
  td->refine_pose = 0;

  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe(sub_image_topic, 10, imageCallback);

  pub = nh.advertise<std_msgs::Bool>(pub_topic_shadow_cam, 1);

  ros::spin();

  apriltag_detector_destroy(td);
  tag16h5_destroy(tf);

  return 0;
}

