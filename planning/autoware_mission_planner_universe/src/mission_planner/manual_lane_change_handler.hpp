// Copyright 2025 Autoware Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MISSION_PLANNER__MANUAL_LANE_CHANGE_HANDLER_HPP_
#define MISSION_PLANNER__MANUAL_LANE_CHANGE_HANDLER_HPP_

#include "service_utils.hpp"

#include <autoware_lanelet2_extension/utility/query.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include <autoware/mission_planner_universe/mission_planner_plugin.hpp>
#include <autoware_internal_debug_msgs/msg/float64_stamped.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <autoware_planning_msgs/srv/set_lanelet_route.hpp>
#include <tier4_planning_msgs/srv/set_preferred_lane.hpp>

#include <optional>
#include <string>

namespace autoware::mission_planner_universe
{

using autoware_planning_msgs::msg::LaneletRoute;
using tier4_planning_msgs::srv::SetPreferredLane;

struct LaneChangeRequestResult
{
  LaneletRoute route;
  bool success;
  std::string message;
};

class ManualLaneChangeHandler : public rclcpp::Node
{
public:
  explicit ManualLaneChangeHandler(const rclcpp::NodeOptions & options);
  void publish_processing_time(autoware_utils::StopWatch<std::chrono::milliseconds> stop_watch)
  {
    autoware_internal_debug_msgs::msg::Float64Stamped processing_time_msg;
    processing_time_msg.stamp = get_clock()->now();
    processing_time_msg.data = stop_watch.toc();
    pub_processing_time_->publish(processing_time_msg);
  }

  void set_preferred_lane(const SetPreferredLane::Request::SharedPtr req, const SetPreferredLane::Response::SharedPtr res);

  LaneChangeRequestResult process_lane_change_request(const int64_t ego_lanelet_id, const SetPreferredLane::Request::SharedPtr req);

  void reset() { original_route_ = std::nullopt; }

private:
  rclcpp::Service<SetPreferredLane>::SharedPtr srv_set_preferred_lane;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odometry_;
  rclcpp::Subscription<LaneletRoute>::SharedPtr sub_route_;
  rclcpp::Publisher<autoware_internal_debug_msgs::msg::Float64Stamped>::SharedPtr
    pub_processing_time_;

  pluginlib::ClassLoader<PlannerPlugin> plugin_loader_;
  std::shared_ptr<PlannerPlugin> planner_;

  nav_msgs::msg::Odometry::ConstSharedPtr odometry_;
  LaneletRoute::ConstSharedPtr current_route_;
  std::function<lanelet::ConstLanelet(const int64_t)> get_lanelet_by_id_;
  std::optional<LaneletRoute::ConstSharedPtr> original_route_;
  rclcpp::Logger logger_;
};

}  // namespace autoware::mission_planner_universe

#endif  // MISSION_PLANNER__MANUAL_LANE_CHANGE_HANDLER_HPP_
