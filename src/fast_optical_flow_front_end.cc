// Copyright 2021 DeepMirror Inc. All rights reserved.
#include "fast_optical_flow_front_end.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

// #include "common/timer.h"
#include "glog/logging.h"
#include "google/protobuf/text_format.h"

namespace dm
{
	namespace vio
	{

		FastOpticalFlowFrontEnd::FastOpticalFlowFrontEnd(
			const dm::Config &config,
			int img_cols,
			int img_rows)
			: kConfig(config),
			  frame_interval_(config.num_cameras * config.frame_interval),
			  latest_point_id_(0),
			  latest_frame_id_(0),
			  last_camera_id_(0),
			  fast_detector_(cv::FastFeatureDetector::create(config.fast_threshold, true)),
			  point_ids_(config.num_cameras, std::vector<uint32_t>()),
			  point_coordinates_(config.num_cameras, std::vector<cv::Point2f>())
		{
			// The camera with the first id is the master camera. We do the tracking from
			// the last frame for the master camera and do tracking from master frame to
			// slave frame for the slave camera. We can have more than one master cameras.
			//! 先考虑单目
			/*
			if (config.num_cameras > 1) {
			  for (int i = 0, iend = config.id_pairs_size; i < iend; ++i) {
				master_camera_ids_.insert(config.id_pairs(i).first());
				paired_master_camera_[config.id_pairs(i).second()] = config.id_pairs(i).first();
				essential_matrix_[config.id_pairs(i).second()] = Eigen::Matrix3d::Zero();
			  }
			} else {
			  master_camera_ids_.insert(0);
			}
			*/
			master_camera_ids_.insert(0);
			if (img_cols > 0 && img_rows > 0)
			{
				image_height_ = img_rows;
				image_width_ = img_cols;
			}
		}

		/*
		FastOpticalFlowFrontEnd::FastOpticalFlowFrontEnd(
			const dm::common::config::VioFrontendConfig& config,
			const std::vector<dm::common::ImageUndistortion>& image_undistortions,
			const std::vector<dm::math::Rigid3d>& poses_camera_in_imu, int img_cols, int img_rows)
			: kConfig(config),
			  kImageUndistortions(image_undistortions),
			  kCameraExtrinsics(poses_camera_in_imu),
			  frame_interval_(config.num_cameras() * config.frame_interval()),
			  latest_point_id_(0),
			  latest_frame_id_(0),
			  last_camera_id_(0),
			  fast_detector_(cv::FastFeatureDetector::create(config.fast_threshold(), true)),
			  point_ids_(config.num_cameras(), std::vector<uint32_t>()),
			  point_coordinates_(config.num_cameras(), std::vector<cv::Point2f>()) {
		  // The camera with the first id is the master camera. We do the tracking from
		  // the last frame for the master camera and do tracking from master frame to
		  // slave frame for the slave camera. We can have more than one master cameras.
		  if (config.num_cameras() > 1) {
			for (int i = 0, iend = config.id_pairs_size(); i < iend; ++i) {
			  master_camera_ids_.insert(config.id_pairs(i).first());
			  paired_master_camera_[config.id_pairs(i).second()] = config.id_pairs(i).first();
			  auto T_slave_master = kCameraExtrinsics[config.id_pairs(i).second()].Inverse() *
									kCameraExtrinsics[config.id_pairs(i).first()];
			  essential_matrix_[config.id_pairs(i).second()] =
				  dm::math::SkewSymmetric(T_slave_master.translation()) *
				  T_slave_master.rotation().toRotationMatrix();
			  slave_tracking_cnt_from_master_[config.id_pairs(i).second()] = 0;
			  paired_cameras_.insert(config.id_pairs(i).first());
			  paired_cameras_.insert(config.id_pairs(i).second());
			}
			uint32_t independent_cam_cnt = 0;
			for (uint32_t i = 0; i < config.num_cameras(); ++i) {
			  slave_tracking_cnt_from_last_[i] = 0;
			  if (!paired_cameras_.count(i)) {
				slave_tracking_cnt_from_last_[i] = ++independent_cam_cnt;
			  }
			}
		  } else {
			master_camera_ids_.insert(0);
		  }
		  if (img_cols > 0 && img_rows > 0) {
			image_height_ = img_rows;
			image_width_ = img_cols;
		  }

		  if (kConfig.num_cameras() > 1 && !kConfig.use_klt_for_slave_tracking()) {
			orb_extractor_ = std::make_shared<dm::vio::ORBextractor>(kConfig.orb_extractor_config());
		  }
		}
		*/

		bool FastOpticalFlowFrontEnd::AddImage(const cv::Mat &image) { return AddImage(0, image); }

		bool FastOpticalFlowFrontEnd::AddImage(const uint32_t camera_id, const cv::Mat &image)
		{
			CHECK_LT(camera_id, kConfig.num_cameras);
			CHECK_EQ(image.type(), CV_8UC1);

			if (latest_frame_id_ == 0 && paired_master_camera_.count(camera_id))
			{
				return false;
			}

			image_height_ = image.rows;
			image_width_ = image.cols;

			// Judging whether the current camera can be a new master camera
			// Notice: Only the camera in the id_pairs can be a new master camera for now.
			//! 只考虑单目
			/*
			if (kConfig.num_cameras > 1 && master_camera_ids_.empty() && paired_cameras_.count(camera_id) &&
				continuous_good_tracking_length_[camera_id] > tracking_length_threshold_) {
			  LOG(WARNING) << "Camera id: " << camera_id << " has become the new master camera";
			  master_camera_ids_.insert(camera_id);

			  uint32_t new_slave_camera_id;
			  if (paired_master_camera_.count(camera_id)) {  // The slave cam becomes the new master cam.
				uint32_t last_master_camera_id = paired_master_camera_.at(camera_id);
				paired_master_camera_.erase(camera_id);
				new_slave_camera_id = last_master_camera_id;
				paired_master_camera_[new_slave_camera_id] = camera_id;
			  } else {  // The origin master cam becomes the new master cam again.
				for (auto pcid : paired_cameras_) {
				  if (pcid != camera_id) {
					new_slave_camera_id = pcid;
					break;
				  }
				}
			  }

			  // Make sure be different with other slave_tracking_cnt_from_last_
			  uint32_t smaller_slave_tracking_cnt_from_last = UINT32_MAX;
			  for (auto s : slave_tracking_cnt_from_last_) {
				if (s.second > 0 && s.second < smaller_slave_tracking_cnt_from_last) {
				  smaller_slave_tracking_cnt_from_last = s.second;
				}
			  }
			  uint32_t min_cam_id_in_paired_cams = UINT32_MAX;
			  for (auto pci : paired_cameras_) {
				if (pci < min_cam_id_in_paired_cams) {
				  min_cam_id_in_paired_cams = pci;
				}
			  }
			  // If the new master camera id is the little one in the paired_cameras_, it means the
			  // slave_tracking_cnt_from_last_ of the slave camera in the paired_cameras_ will be added once
			  // in the next image processing. So we subtract once now.
			  if (camera_id == min_cam_id_in_paired_cams || !backend_is_initialized_) {
				--smaller_slave_tracking_cnt_from_last;
			  }
			  slave_tracking_cnt_from_last_[new_slave_camera_id] = smaller_slave_tracking_cnt_from_last;
			  // Keep the slave_tracking_cnt_from_master_[new_slave_camera_id] is equal to
			  // or a multiple of slave_tracking_cnt_from_last_[new_slave_camera_id].
			  slave_tracking_cnt_from_master_[new_slave_camera_id] = smaller_slave_tracking_cnt_from_last;
			  auto T_slave_master =
				  kCameraExtrinsics[new_slave_camera_id].Inverse() * kCameraExtrinsics[camera_id];
			  essential_matrix_[new_slave_camera_id] = dm::math::SkewSymmetric(T_slave_master.translation()) *
													   T_slave_master.rotation().toRotationMatrix();
			}
			*/

			cv::Mat equalized_image;
			if (kConfig.do_equalization)
			{
				cv::equalizeHist(image, equalized_image);
			}

			std::vector<cv::Mat> cur_pyramid;
			cv::buildOpticalFlowPyramid(kConfig.do_equalization ? equalized_image : image, cur_pyramid,
										cv::Size(kConfig.window_size, kConfig.window_size),
										kConfig.pyramid_levels);

			if (grids_.size() == 0)
			{
				grids_.resize(std::ceil(1.0 * image_width_ / grid_size_));
				for (int i = 0, iend = std::ceil(1.0 * image_width_ / grid_size_); i < iend; ++i)
				{
					grids_[i].resize(std::ceil(1.0 * image_height_ / grid_size_));
				}
			}
			// First we do the tracking from the main camera to the slave camera if the camera is slave.
			//! 单目不进这个if
			/*
			if (!master_camera_ids_.empty() && paired_master_camera_.count(camera_id))
			{
				CHECK(essential_matrix_.count(camera_id));
				if (!backend_is_initialized_ || (slave_tracking_cnt_from_master_[camera_id]++ %
													 kConfig.slave_tracking_interval_from_master ==
												 0))
				{
					kConfig.use_klt_for_slave_tracking
						? DoTrackingFromMasterWithKLT(camera_id, cur_pyramid)
						: DoTrackingFromMasterWithDescriptorMatching(camera_id, cur_pyramid);
				}
				GenerateMask(camera_id, &temporary_pts_tracked_from_master_,
							 &temporary_ids_tracked_from_master_, &temporary_lengths_tracked_from_master_);
			}
			*/

			std::vector<cv::Point2f> cur_pts; // 存储当前帧特征点
			// Then we do the tracking from the last frame to the current frame for one camera.
			if (point_coordinates_[camera_id].size() > 0)
			{ // 非第一帧
				if (master_camera_ids_.count(camera_id) || master_camera_ids_.empty() ||
					!backend_is_initialized_ ||
					(slave_tracking_cnt_from_last_[camera_id]++ % kConfig.slave_tracking_interval_from_last ==
					 0))
				{ // 始终为true
					DoTrackingFromLastFrame(camera_id, cur_pyramid, &cur_pts);

					DetectFastFeatures(camera_id, cur_pyramid[0], &cur_pts);

					last_pyramid_[camera_id] = std::move(cur_pyramid);
					point_coordinates_[camera_id] = std::move(cur_pts);
					last_is_processed_[camera_id] = true;
				}
				else
				{
					last_is_processed_[camera_id] = false;
				}
			}
			else
			{ // 第一帧
				if (paired_master_camera_.count(camera_id))
				{
					// The process for the slave camera in the first time.
					cur_pts = temporary_pts_tracked_from_master_;
					point_ids_[camera_id] = temporary_ids_tracked_from_master_;
					tracking_lengths_[camera_id] = temporary_lengths_tracked_from_master_;
				}
				// For complete occlusion case. We add tracking_cnt even if no points are tracked.
				if (latest_frame_id_ >= kConfig.num_cameras && backend_is_initialized_ &&
					!master_camera_ids_.empty())
				{
					++slave_tracking_cnt_from_last_[camera_id];
				}
				DetectFastFeatures(camera_id, cur_pyramid[0], &cur_pts);
				last_pyramid_[camera_id] = std::move(cur_pyramid);
				point_coordinates_[camera_id] = std::move(cur_pts);
				last_is_processed_[camera_id] = true;
			}

			last_camera_id_ = camera_id;

			//! 单目没有paired_cameras_
			if (last_is_processed_[camera_id] && paired_cameras_.count(camera_id))
			{
				if (IsGoodTracking(camera_id))
				{
					continuous_good_tracking_length_[camera_id]++;
					continuous_bad_tracking_length_[camera_id] = 0;
				}
				else
				{
					continuous_bad_tracking_length_[camera_id]++;
					continuous_good_tracking_length_[camera_id] = 0;
				}
			}

			//! 单目num_cameras = 1
			if (kConfig.num_cameras > 1 && master_camera_ids_.count(camera_id) &&
				continuous_bad_tracking_length_[camera_id] > tracking_length_threshold_)
			{
				LOG(WARNING) << "Camera id: " << camera_id
							 << " bad tracking, will change master camera to others later";
				master_camera_ids_.erase(camera_id);
				for (auto slave_master : paired_master_camera_)
				{
					if (slave_master.second == camera_id)
					{
						slave_tracking_cnt_from_last_[slave_master.first] = 0;
						slave_tracking_cnt_from_master_[slave_master.first] = 0;
					}
				}
				temporary_pts_tracked_from_master_.clear();
				temporary_ids_tracked_from_master_.clear();
				temporary_lengths_tracked_from_master_.clear();
				mask_[camera_id] = cv::Mat(image_height_, image_width_, CV_8UC1, cv::Scalar(255));
				for (int i = 0; i < grids_.size(); ++i)
				{
					for (int j = 0; j < grids_[i].size(); ++j)
					{
						grids_[i][j].clear();
					}
				}
			}

			// We send the tracking result from the frondend to the backend every frame_interval_ frames.
			if (latest_frame_id_ < frame_interval_)
			{
				++latest_frame_id_;
				return false;
			}
			else
			{
				return latest_frame_id_++ % frame_interval_ < kConfig.num_cameras;
			}
		}

		/*void FastOpticalFlowFrontEnd::GetLatestVioFrame(dm::vio::VioProblemFrame* frame) {
		  frame->camera_id = last_camera_id_;
		  if (last_is_processed_.at(last_camera_id_)) {
			if (point_coordinates_[last_camera_id_].size() > 0) {
			  frame->point_observations.resize(point_coordinates_[last_camera_id_].size());
			  for (size_t i = 0, iend = point_coordinates_[last_camera_id_].size(); i < iend; ++i) {
				frame->point_observations[i].point_id = point_ids_[last_camera_id_][i];
				frame->point_observations[i].x = point_coordinates_[last_camera_id_][i].x;
				frame->point_observations[i].y = point_coordinates_[last_camera_id_][i].y;
			  }
			}
		  }
		}
		*/

		bool FastOpticalFlowFrontEnd::IsGoodTracking(uint32_t cam_id)
		{
			// We think it is a good tracking if there are many points which has long tracking
			if (point_coordinates_[cam_id].size() > kConfig.target_num_of_features / 2)
			{
				uint32_t cnt = 0;
				uint32_t max_tracking_length = 0;
				for (auto tl : tracking_lengths_[cam_id])
				{
					if (tl > tracking_length_threshold_)
					{
						cnt++;
					}
					if (tl > max_tracking_length)
					{
						max_tracking_length = tl;
					}
				}
				if (max_tracking_length < 20 || cnt > kConfig.target_num_of_features / 3)
				{
					return true;
				}
			}
			return false;
			// Todo(chaizheng): Judging whether it is evenly distributed
		}

		bool FastOpticalFlowFrontEnd::InBorder(const cv::Point2f &pt, const uint32_t borderSize) const
		{
#ifdef __EMSCRIPTEN__
			uint32_t img_x = cv::cvRound(pt.x);
			uint32_t img_y = cv::cvRound(pt.y);
#else
			uint32_t img_x = cvRound(pt.x);
			uint32_t img_y = cvRound(pt.y);
#endif
			return borderSize <= img_x && img_x < image_width_ - borderSize && borderSize <= img_y &&
				   img_y < image_height_ - borderSize;
		}

		template <typename T>
		void FastOpticalFlowFrontEnd::CombineVector(const std::vector<T> &src_v1,
													const std::vector<T> &src_v2, std::vector<T> *dst_v)
		{
			dst_v->clear();
			dst_v->insert(dst_v->end(), src_v1.begin(), src_v1.end());
			dst_v->insert(dst_v->end(), src_v2.begin(), src_v2.end());
		}

		int FastOpticalFlowFrontEnd::FindIdIndex(const std::vector<uint32_t> &ids,
												 const uint32_t &id_to_be_found) const
		{
			for (size_t i = 0, iend = ids.size(); i < iend; ++i)
			{
				if (ids[i] == id_to_be_found)
				{
					return static_cast<int>(i);
				}
			}
			return -1;
		}

		void FastOpticalFlowFrontEnd::GenerateMask(const uint32_t &cam_id, std::vector<cv::Point2f> *pts,
												   std::vector<uint32_t> *ids,
												   std::vector<uint32_t> *lengths)
		{
			cv::Mat mask(image_height_, image_width_, CV_8UC1, cv::Scalar(255));

			std::vector<std::pair<int, std::pair<cv::Point2f, int>>> lengths_pts_ids(pts->size());
			for (size_t i = 0, iend = pts->size(); i < iend; ++i)
			{
				lengths_pts_ids[i] = std::make_pair((*lengths)[i], std::make_pair((*pts)[i], (*ids)[i]));
			}

			std::sort(lengths_pts_ids.begin(), lengths_pts_ids.end(),
					  [](const std::pair<int, std::pair<cv::Point2f, int>> &a,
						 const std::pair<int, std::pair<cv::Point2f, int>> &b)
					  {
						  return a.first == b.first ? a.second.second < b.second.second : a.first > b.first;
					  });

			pts->clear();
			ids->clear();
			lengths->clear();
			for (const auto &it : lengths_pts_ids)
			{
				if (mask.at<uchar>(it.second.first) == 255)
				{
					pts->emplace_back(it.second.first);
					ids->emplace_back(it.second.second);
					lengths->emplace_back(it.first);
					cv::circle(mask, it.second.first, kConfig.minimum_distance, 0, -1);
				}
			}
			mask_[cam_id] = mask;
		}

		double FastOpticalFlowFrontEnd::CalcDistance(const cv::Point2f &pt1, const cv::Point2f &pt2) const
		{
			double dx = pt1.x - pt2.x;
			double dy = pt1.y - pt2.y;
			return sqrt(dx * dx + dy * dy);
		}

		void FastOpticalFlowFrontEnd::DetectFastFeatures(const uint32_t &cam_id, const cv::Mat &img,
														 std::vector<cv::Point2f> *cur_pts)
		{
			if (cur_pts->size() > kConfig.target_num_of_features * kConfig.detect_ratio_threshold)
			{
				return;
			}

			GenerateMask(cam_id, cur_pts, &point_ids_[cam_id], &tracking_lengths_[cam_id]);

			std::vector<cv::KeyPoint> fast_features;
			fast_detector_->detect(img, fast_features, mask_[cam_id]);
			for (size_t i = 0, iend = fast_features.size(); i < iend; ++i)
			{
				fast_features[i].class_id = i;
			}

			std::sort(fast_features.begin(), fast_features.end(),
					  [](const cv::KeyPoint &a, const cv::KeyPoint &b)
					  {
						  return a.response == b.response ? a.class_id < b.class_id : a.response > b.response;
					  });

			size_t need_cnt = kConfig.target_num_of_features - cur_pts->size();
			for (size_t i = 0, iend = fast_features.size(), feature_cnt = 0;
				 i < iend && feature_cnt < need_cnt; ++i)
			{
				cv::Point2f pt = fast_features[i].pt;
				if (mask_[cam_id].at<uchar>(pt) == 0)
				{
					continue;
				}
				cur_pts->emplace_back(pt);
				point_ids_[cam_id].emplace_back(latest_point_id_++);
				tracking_lengths_[cam_id].emplace_back(1);
				++feature_cnt;
				cv::circle(mask_[cam_id], pt, kConfig.minimum_distance, cv::Scalar(0), -1);
			}
		}

		//! 单目不考虑这个
		/*
		void FastOpticalFlowFrontEnd::DoTrackingFromMasterWithKLT(const uint32_t& slave_cam_id,
																  const std::vector<cv::Mat>& cur_pyramid) {
		  if (temporary_pts_tracked_from_master_.size() > 0) {
			temporary_pts_tracked_from_master_.clear();
			temporary_ids_tracked_from_master_.clear();
			temporary_lengths_tracked_from_master_.clear();
		  }
		  CHECK(paired_master_camera_.count(slave_cam_id));
		  uint32_t master_cam_id = paired_master_camera_[slave_cam_id];
		  std::vector<double> init_depths = {0.5, 2};
		  std::vector<uchar> status;
		  std::vector<float> err;
		  cv::TermCriteria term_criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

		  for (size_t i = 0, iend = point_coordinates_[master_cam_id].size(); i < iend; ++i) {
			for (size_t j = 0; j < init_depths.size(); ++j) {
			  cv::Point2f undis_ray_0 =
				  kImageUndistortions[master_cam_id].LiftPointToRay(point_coordinates_[master_cam_id][i]);
			  Eigen::Vector3d pi = kCameraExtrinsics[master_cam_id] *
								   (init_depths[j] * Eigen::Vector3d(undis_ray_0.x, undis_ray_0.y, 1));
			  Eigen::Vector3d pc1 = kCameraExtrinsics[slave_cam_id].Inverse() * pi;
			  Eigen::Vector2d predict_uv =
				  kImageUndistortions[slave_cam_id].ProjectRayToPoint(pc1.head(2) / pc1[2]);
			  std::vector<cv::Point2f> tracking_uv = {cv::Point2f(predict_uv[0], predict_uv[1])};

			  if (InBorder(tracking_uv[0])) {
				cv::calcOpticalFlowPyrLK(last_pyramid_[master_cam_id], cur_pyramid,
										 std::vector<cv::Point2f>{point_coordinates_[master_cam_id][i]},
										 tracking_uv, status, err, cv::Size(7, 7), 1, term_criteria,
										 cv::OPTFLOW_USE_INITIAL_FLOW);
				if (status[0] && InBorder(tracking_uv[0])) {
				  cv::Point2f undis_ray_1 =
					  kImageUndistortions[slave_cam_id].LiftPointToRay(tracking_uv[0]);
				  double epl_err = Eigen::Vector3d(undis_ray_1.x, undis_ray_1.y, 1).transpose() *
								   essential_matrix_[slave_cam_id] *
								   Eigen::Vector3d(undis_ray_0.x, undis_ray_0.y, 1);
				  if (fabs(epl_err * kImageUndistortions[slave_cam_id].fx()) <
					  kConfig.epipolar_threshold()) {
					temporary_pts_tracked_from_master_.emplace_back(tracking_uv[0]);
					temporary_ids_tracked_from_master_.emplace_back(point_ids_[master_cam_id][i]);
					temporary_lengths_tracked_from_master_.emplace_back(
						tracking_lengths_[master_cam_id][i]);
					break;
				  }
				}
			  }
			}
		  }
		}
		*/

		//! 单目不考虑这个
		/*
		void FastOpticalFlowFrontEnd::DoTrackingFromMasterWithDescriptorMatching(
			const uint32_t& slave_cam_id, const std::vector<cv::Mat>& cur_pyramid) {
		  if (temporary_pts_tracked_from_master_.size() > 0) {
			temporary_pts_tracked_from_master_.clear();
			temporary_ids_tracked_from_master_.clear();
			temporary_lengths_tracked_from_master_.clear();
		  }
		  CHECK(paired_master_camera_.count(slave_cam_id));
		  uint32_t master_cam_id = paired_master_camera_[slave_cam_id];
		  std::vector<double> init_depths = {0.1, 100.0};

		  std::vector<cv::KeyPoint> kps_master;
		  kps_master.reserve(point_coordinates_[master_cam_id].size());
		  std::vector<std::pair<uint32_t, uint32_t>> ids_lengths;
		  ids_lengths.reserve(point_coordinates_[master_cam_id].size());
		  for (size_t i = 0, iend = point_coordinates_[master_cam_id].size(); i < iend; ++i) {
			if (InBorder(point_coordinates_[master_cam_id][i], 16)) {
			  kps_master.emplace_back(cv::KeyPoint(point_coordinates_[master_cam_id][i], 1));
			  ids_lengths.emplace_back(
				  std::make_pair(point_ids_[master_cam_id][i], tracking_lengths_[master_cam_id][i]));
			}
		  }
		  cv::Mat desc_master;
		  orb_extractor_->GetDescriptors(last_pyramid_[master_cam_id][0], &kps_master, &desc_master);

		  std::vector<cv::KeyPoint> kps_slave;
		  cv::Mat desc_slave;
		  orb_extractor_->DetectAndCompute(cur_pyramid[0], &kps_slave, &desc_slave);

		  std::vector<cv::Point2f> pts(kps_slave.size());
		  for (size_t i = 0, iend = kps_slave.size(); i < iend; ++i) {
			pts[i] = kps_slave[i].pt;
			grids_[std::floor(pts[i].x / grid_size_)][std::floor(pts[i].y / grid_size_)].emplace_back(i);
		  }
		  std::vector<cv::Point2f> undis_rays_1 = kImageUndistortions[slave_cam_id].LiftPointsToRays(pts);

		  for (size_t i = 0, iend = kps_master.size(); i < iend; ++i) {
			cv::Point2f undis_ray_0 = kImageUndistortions[master_cam_id].LiftPointToRay(kps_master[i].pt);
			std::vector<cv::Point2f> search_area_pts(2);
			for (size_t j = 0; j < init_depths.size(); ++j) {
			  Eigen::Vector3d pi = kCameraExtrinsics[master_cam_id] *
								   (init_depths[j] * Eigen::Vector3d(undis_ray_0.x, undis_ray_0.y, 1));
			  Eigen::Vector3d pc1 = kCameraExtrinsics[slave_cam_id].Inverse() * pi;
			  Eigen::Vector2d predict_uv =
				  kImageUndistortions[slave_cam_id].ProjectRayToPoint(pc1.head(2) / pc1[2]);
			  predict_uv[0] =
				  predict_uv[0] < 0 ? 0 : (predict_uv[0] > image_width_ ? image_width_ : predict_uv[0]);
			  predict_uv[1] =
				  predict_uv[1] < 0 ? 0 : (predict_uv[1] > image_height_ ? image_height_ : predict_uv[1]);
			  search_area_pts[j] = cv::Point2f(predict_uv[0], predict_uv[1]);
			}
			// Continue if the search area is too small.
			if (cv::norm(search_area_pts[0] - search_area_pts[1]) < 20) {
			  continue;
			}

			int search_grid_min_x =
				std::floor(std::min(search_area_pts[0].x, search_area_pts[1].x) / grid_size_);
			int search_grid_min_y =
				std::floor(std::min(search_area_pts[0].y, search_area_pts[1].y) / grid_size_);
			int search_grid_max_x =
				std::ceil(std::max(search_area_pts[0].x, search_area_pts[1].x) / grid_size_);
			int search_grid_max_y =
				std::ceil(std::max(search_area_pts[0].y, search_area_pts[1].y) / grid_size_);

			int min_distance = 255;
			int match_id = -1;
			for (size_t x = search_grid_min_x; x < search_grid_max_x; ++x) {
			  for (size_t y = search_grid_min_y; y < search_grid_max_y; ++y) {
				for (const auto id : grids_[x][y]) {
				  double epl_err = Eigen::Vector3d(undis_rays_1[id].x, undis_rays_1[id].y, 1).transpose() *
								   essential_matrix_[slave_cam_id] *
								   Eigen::Vector3d(undis_ray_0.x, undis_ray_0.y, 1);
				  if (fabs(epl_err * kImageUndistortions[slave_cam_id].fx()) <
					  kConfig.epipolar_threshold()) {
					int distance =
						orb_extractor_->DescriptorDistance(desc_master.row(i), desc_slave.row(id));
					if (distance < 100 && distance < min_distance) {
					  min_distance = distance;
					  match_id = id;
					}
				  }
				}
			  }
			}
			if (match_id >= 0) {
			  temporary_pts_tracked_from_master_.emplace_back(kps_slave[match_id].pt);
			  temporary_ids_tracked_from_master_.emplace_back(ids_lengths[i].first);
			  temporary_lengths_tracked_from_master_.emplace_back(ids_lengths[i].second);
			}
		  }

		  for (int i = 0; i < grids_.size(); ++i) {
			for (int j = 0; j < grids_[i].size(); ++j) {
			  grids_[i][j].clear();
			}
		  }
		}
		*/

		void FastOpticalFlowFrontEnd::DoTrackingFromLastFrame(const uint32_t &cam_id,
															  const std::vector<cv::Mat> &cur_pyramid,
															  std::vector<cv::Point2f> *cur_pts)
		{
			std::vector<uchar> status;
			std::vector<float> err;
			cv::calcOpticalFlowPyrLK(last_pyramid_[cam_id], cur_pyramid, point_coordinates_[cam_id], *cur_pts,
									 status, err, cv::Size(kConfig.window_size, kConfig.window_size),
									 kConfig.pyramid_levels);

			// TODO(chaizheng): Do the ransac to reject outliers.

			// Reverse tracking check
			std::vector<uchar> reverse_status;
			std::vector<cv::Point2f> reverse_pts;
			if (kConfig.do_reverse_check)
			{
				reverse_pts = point_coordinates_[cam_id];
				cv::calcOpticalFlowPyrLK(
					cur_pyramid, last_pyramid_[cam_id], *cur_pts, reverse_pts, reverse_status, err,
					cv::Size(kConfig.window_size, kConfig.window_size), kConfig.pyramid_levels,
					cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
					cv::OPTFLOW_USE_INITIAL_FLOW);
			}
			// We will do the mergeing process if the camera is a slave camera.
			//! 单目不进这个判断
			if (paired_master_camera_.count(cam_id))
			{
				std::vector<cv::Point2f> temporary_pts_tracked_from_last_slave_frame;
				std::vector<uint32_t> temporary_ids_tracked_from_last_slave_frame;
				std::vector<uint32_t> temporary_lengths_tracked_from_last_slave_frame;
				for (size_t i = 0, iend = status.size(); i < iend; ++i)
				{
					if (status[i] && InBorder((*cur_pts)[i]) &&
						(!kConfig.do_reverse_check ||
						 (reverse_status[i] && CalcDistance(point_coordinates_[cam_id][i], reverse_pts[i]) <
												   kConfig.max_reverse_check_distance)))
					{
						int id_pos = FindIdIndex(temporary_ids_tracked_from_master_, point_ids_[cam_id][i]);
						if (id_pos >= 0)
						{
							if (CalcDistance((*cur_pts)[i], temporary_pts_tracked_from_master_[id_pos]) >
								kConfig.max_master_slave_tracking_distance)
							{
								// If the distance is large, we trust the tracking from the slave camera,
								// and replace the point which is tracked from the master camera.
								temporary_pts_tracked_from_master_[id_pos] = (*cur_pts)[i];
							}
						}
						else if (mask_[cam_id].at<uchar>((*cur_pts)[i]) == 255)
						{
							// For an independent point tracked from the last slave frame, we add it if there is
							// no point tracked from the master around.
							temporary_pts_tracked_from_last_slave_frame.emplace_back((*cur_pts)[i]);
							temporary_ids_tracked_from_last_slave_frame.emplace_back(point_ids_[cam_id][i]);
							temporary_lengths_tracked_from_last_slave_frame.emplace_back(
								tracking_lengths_[cam_id][i] + 1);
						}
					}
				}
				CombineVector(temporary_pts_tracked_from_master_, temporary_pts_tracked_from_last_slave_frame,
							  cur_pts);
				CombineVector(temporary_ids_tracked_from_master_, temporary_ids_tracked_from_last_slave_frame,
							  &point_ids_[cam_id]);
				CombineVector(temporary_lengths_tracked_from_master_,
							  temporary_lengths_tracked_from_last_slave_frame, &tracking_lengths_[cam_id]);
				temporary_pts_tracked_from_master_.clear();
				temporary_ids_tracked_from_master_.clear();
				temporary_lengths_tracked_from_master_.clear();
				mask_[cam_id] = cv::Mat(image_height_, image_width_, CV_8UC1, cv::Scalar(255));
			}
			else
			{
				size_t keep_index = 0;
				for (size_t i = 0, iend = status.size(); i < iend; ++i)
				{
					if (status[i] && InBorder((*cur_pts)[i]) &&
						(!kConfig.do_reverse_check ||
						 (reverse_status[i] && CalcDistance(point_coordinates_[cam_id][i], reverse_pts[i]) <
												   kConfig.max_reverse_check_distance)))
					{
						(*cur_pts)[keep_index] = (*cur_pts)[i];
						point_ids_[cam_id][keep_index] = point_ids_[cam_id][i];
						tracking_lengths_[cam_id][keep_index++] = tracking_lengths_[cam_id][i] + 1;
					}
				}
				(*cur_pts).resize(keep_index);
				point_ids_[cam_id].resize(keep_index);
				tracking_lengths_[cam_id].resize(keep_index);
			}
		}

		std::vector<double> FastOpticalFlowFrontEnd::GetTimeCosts()
		{
			std::vector<double> cost(4);
			cost[0] = total_time_ / latest_frame_id_;
			cost[1] = buildpyramid_time_ / latest_frame_id_;
			cost[2] = tracking_time_ / latest_frame_id_;
			cost[3] = detection_time_ / latest_frame_id_;
			return cost;
		}

	} // namespace vio
} // namespace dm
