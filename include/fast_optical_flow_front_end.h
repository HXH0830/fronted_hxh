// Copyright 2021 DeepMirror Inc. All rights reserved.

#ifndef VIO_FRONTEND_FAST_OPTICAL_FLOW_FRONT_END_H_
#define VIO_FRONTEND_FAST_OPTICAL_FLOW_FRONT_END_H_

#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

// #include "common/image_undistortion.h"
// #include "math/rigid.h"
// #include "math/transformation_utils.h"
// #include "proto/vio_frontend_config.pb.h"
// #include "vio/data_types.h"
// #include "vio/frontend/orb_extractor.h"

namespace dm {
struct Config{
	int target_num_of_features = 150;
	double detect_ratio_threshold = 1.0;
	int fast_threshold = 15;
	int minimum_distance = 30;
	int window_size = 21;
	int pyramid_levels = 3;
	bool do_equalization = true;
	bool do_reverse_check = true;
	double max_reverse_check_distance = 0.5;
	int frame_interval = 1;
	int num_cameras = 1;//先尝试单目
	struct IdPairs {
		int first = 0;
		int second = 1;
	};
	IdPairs id_pairs;
	float max_master_slave_tracking_distance = 1.0;
	bool use_klt_for_slave_tracking = true;
	float epipolar_threshold = 0.1;
	struct OrbExtractorConfig {
		int features_num = 1200;
		float pyramid_scale_factor = 1.2;
		int pyramid_levels = 2;
		int init_th_fast = 20;
		int min_th_fast = 7;
	};
	OrbExtractorConfig orb_extractor_config;
	int slave_tracking_interval_from_last = 1;
	int slave_tracking_interval_from_master = 1;
};
namespace vio
{

  class FastOpticalFlowFrontEnd
  {
  public:
    FastOpticalFlowFrontEnd() = delete;
	//构造函数去掉了去畸变相关的参数
    FastOpticalFlowFrontEnd(const dm::Config &config,
                            int img_cols = -1, int img_rows = -1);
    // FastOpticalFlowFrontEnd(const dm::common::config::VioFrontendConfig& config,
    //                         const std::vector<dm::common::ImageUndistortion>& image_undistortions,
    //                         const std::vector<dm::math::Rigid3d>& poses_camera_in_imu,
    //                         int img_cols = -1, int img_rows = -1);

    // Add image from 0-th camera. Mostly used for single camera case.
    // Returns true if the frame is detected as a keyframe.
    bool AddImage(const cv::Mat &image);

    // Add image from camera with id of camera_id.
    // Returns true if the frame is to be sent to the backend.
    bool AddImage(const uint32_t camera_id, const cv::Mat &image);

    std::vector<double> GetTimeCosts();

    // void GetLatestVioFrame(dm::vio::VioProblemFrame* frame);

    const std::vector<std::vector<uint32_t>> &point_ids() const { return point_ids_; }
    const std::vector<std::vector<cv::Point2f>> &point_coordinates() const
    {
      return point_coordinates_;
    }

    const std::set<uint32_t> &master_camera_ids() { return master_camera_ids_; }
    const std::set<uint32_t> &paired_cameras() { return paired_cameras_; }
    const std::unordered_map<uint32_t, bool> &last_is_processed() { return last_is_processed_; }
    void set_latest_point_id(uint32_t pt_id) { latest_point_id_ = pt_id; }
    uint32_t latest_point_id() { return latest_point_id_; }
    void SetStatus(bool backend_is_initialized) { backend_is_initialized_ = backend_is_initialized; }

  private:
    bool InBorder(const cv::Point2f &pt, const uint32_t borderSize = 1U) const;
    template <typename T>
    void CombineVector(const std::vector<T> &src_v1, const std::vector<T> &src_v2,
                       std::vector<T> *dst_v);
    int FindIdIndex(const std::vector<uint32_t> &ids, const uint32_t &id_to_be_found) const;
    // We generate the mask by track length in descending order.
    void GenerateMask(const uint32_t &cam_id, std::vector<cv::Point2f> *pts,
                      std::vector<uint32_t> *ids, std::vector<uint32_t> *lengths);
    double CalcDistance(const cv::Point2f &pt1, const cv::Point2f &pt2) const;
    void DetectFastFeatures(const uint32_t &cam_id, const cv::Mat &img,
                            std::vector<cv::Point2f> *cur_pts);
    // If a camera is slave, we do the tracking from its master camera.
    //! 单目暂时不需要
	/*
	void DoTrackingFromMasterWithKLT(const uint32_t &slave_cam_id,
                                     const std::vector<cv::Mat> &cur_pyramid);
    void DoTrackingFromMasterWithDescriptorMatching(const uint32_t &slave_cam_id,
                                                    const std::vector<cv::Mat> &cur_pyramid);
    */
	// Do the tracking from the last frame to the current frame for one camera.
    void DoTrackingFromLastFrame(const uint32_t &cam_id, const std::vector<cv::Mat> &cur_pyramid,
                                 std::vector<cv::Point2f> *cur_pts);

    // Judge if the current frame has good tracking from the last frame.
    bool IsGoodTracking(uint32_t cam_id);



	// The config contains the parameters about extracting features, tracking features and
    // processing image. See vio_frontend_config.proto for more details
    const dm::Config kConfig;
    // Lift points to rays or project rays to points with undistortion.
    //const std::vector<dm::common::ImageUndistortion> kImageUndistortions;
	//! 单目先注释掉这个
    //const std::vector<dm::math::Rigid3d> kCameraExtrinsics;
    // E21
    std::map<uint32_t, Eigen::Matrix3d> essential_matrix_;
    // How many frames to skip before sending to backend.
    uint32_t frame_interval_;
    // Which camera is the master camera paired up with the current camera.
    std::unordered_map<uint32_t, uint32_t> paired_master_camera_;
    std::set<uint32_t> master_camera_ids_;
    uint32_t latest_point_id_;
    // We drop the first image for stable initialization with more imu measurements.
    uint32_t latest_frame_id_;
    uint32_t last_camera_id_;
    cv::Ptr<cv::FastFeatureDetector> fast_detector_;
    // Point ids and point coordinates from the last gray level image for each camera.
    std::vector<std::vector<uint32_t>> point_ids_;
    std::vector<std::vector<cv::Point2f>> point_coordinates_;
    // We get the temporary points and their ids and lengths which tracked from the master camera for
    // the slave camera, and they will be merged with that tracked from the last slave camera later.
    std::vector<cv::Point2f> temporary_pts_tracked_from_master_;
    std::vector<uint32_t> temporary_ids_tracked_from_master_;
    std::vector<uint32_t> temporary_lengths_tracked_from_master_;
    // Optical flow pyramid constructed from the last frame for each camera.
    std::unordered_map<uint32_t, std::vector<cv::Mat>> last_pyramid_;
    // The masks help to keep the points evenly distributed.
    std::unordered_map<uint32_t, cv::Mat> mask_;
    uint32_t image_height_;
    uint32_t image_width_;
    // The tracking lengths of points, and we will generate the mask by the tracking lengths in
    // descending order.
    std::unordered_map<uint32_t, std::vector<uint32_t>> tracking_lengths_;

    double buildpyramid_time_ = 0.0;
    double tracking_time_ = 0.0;
    double detection_time_ = 0.0;
    double total_time_ = 0.0;

    // std::shared_ptr<dm::vio::ORBextractor> orb_extractor_;
    std::vector<std::vector<std::vector<size_t>>> grids_;
    int grid_size_ = 20;

    bool backend_is_initialized_ = false;
    // The master camera can be only generated in the paired_cams (id_pairs).
    std::set<uint32_t> paired_cameras_;
    // The count of the slave camera tracked from the last frame, is used to skip.
    // The size of this map is equal to the size of total cameras.
    std::unordered_map<uint32_t, uint32_t> slave_tracking_cnt_from_last_;
    // The count of the slave camera tracked from the master frame, is used to skip.
    // The size of this map is equal to the size of id_pairs.
    std::unordered_map<uint32_t, uint32_t> slave_tracking_cnt_from_master_;
    // Will be true if do tracking for cur frame. For drawing images out of here.
    std::unordered_map<uint32_t, bool> last_is_processed_;
    // The count of IsGoodTracking(), to switch between master and slave cameras.
    std::unordered_map<uint32_t, uint32_t> continuous_good_tracking_length_;
    // The count of !IsGoodTracking(), to switch between master and slave cameras.
    std::unordered_map<uint32_t, uint32_t> continuous_bad_tracking_length_;
    uint32_t tracking_length_threshold_ = 3;
  };

} // namespace vio
}  // namespace dm

#endif  // VIO_FRONTEND_FAST_OPTICAL_FLOW_FRONT_END_H_
