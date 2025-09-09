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

#include <autoware/mission_planner_universe/mission_planner_plugin.hpp>
#include <autoware_lanelet2_extension/utility/query.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>

#include <autoware_internal_debug_msgs/msg/float64_stamped.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <tier4_planning_msgs/srv/manual_lane_change_request.hpp>

#include <optional>
#include <string>

namespace autoware::mission_planner_universe
{

using autoware_planning_msgs::msg::LaneletRoute;
using tier4_planning_msgs::srv::ManualLaneChangeRequest;

struct LaneChangeRequestResult
{
  LaneletRoute route;
  bool success;
  std::string message;
};

enum class DIRECTION {
  MANUAL_LEFT,
  MANUAL_RIGHT,
  AUTO,
};

class ManualLaneChangeHandler : public rclcpp::Node
{
public:
  explicit ManualLaneChangeHandler(const rclcpp::NodeOptions & options)
  : Node("manual_lane_change_handler", options)
  {
    auto planner = plugin_loader_.createSharedInstance(
      "autoware::mission_planner_universe::lanelet2::DefaultPlanner");
    planner->initialize(this);

    get_lanelet_by_id_ = [&](const int64_t id) {
      return planner->getRouteHandler().getLaneletMapPtr()->laneletLayer.get(id);
    };

    srv_manual_lane_change_request_ = create_service<ManualLaneChangeRequest>(
      "~/manual_lane_change_request",
      service_utils::handle_exception(
        &ManualLaneChangeHandler::on_manual_lane_change_request, this));

    pub_processing_time_ =
      this->create_publisher<autoware_internal_debug_msgs::msg::Float64Stamped>(
        "~/debug/processing_time_ms", 1);
  }

  void on_manual_lane_change_request(
    const ManualLaneChangeRequest::Request::SharedPtr req,
    const ManualLaneChangeRequest::Response::SharedPtr res)
  {
    if (req->code == 3) {  // RESET
      reset();
      res->status.success = true;
      res->status.message = "Manual lane change handler is reset.";
      res->new_route = LaneletRoute();
      return;
    }

    current_route_ =
      std::make_shared<const autoware_planning_msgs::msg::LaneletRoute>(req->current_route);
    const DIRECTION override_direction = req->code == 0   ? DIRECTION::MANUAL_LEFT
                                         : req->code == 1 ? DIRECTION::MANUAL_RIGHT
                                                          : DIRECTION::AUTO;
    const auto lane_change_request_result =
      process_lane_change_request(req->lane_id, override_direction);
    res->status.message = lane_change_request_result.message;
    res->status.success = lane_change_request_result.success;
    res->new_route = std::move(lane_change_request_result.route);
  }

  LaneChangeRequestResult process_lane_change_request(
    const int64_t ego_lanelet_id, const DIRECTION override_direction);

  void reset() { original_route_ = std::nullopt; }
  void publish_processing_time(autoware_utils::StopWatch<std::chrono::milliseconds> stop_watch);

private:
  pluginlib::ClassLoader<PlannerPlugin> plugin_loader_{
    "autoware_mission_planner_universe", "autoware::mission_planner_universe::PlannerPlugin"};
  std::optional<LaneletRoute::ConstSharedPtr> original_route_{std::nullopt};
  LaneletRoute::ConstSharedPtr current_route_{nullptr};
  std::function<lanelet::ConstLanelet(const int64_t)> get_lanelet_by_id_;
  rclcpp::Logger logger_{rclcpp::get_logger("ManualLaneChangeHandler")};
  rclcpp::Service<ManualLaneChangeRequest>::SharedPtr srv_manual_lane_change_request_;

  rclcpp::Publisher<autoware_internal_debug_msgs::msg::Float64Stamped>::SharedPtr
    pub_processing_time_;
};

}  // namespace autoware::mission_planner_universe

#endif  // MISSION_PLANNER__MANUAL_LANE_CHANGE_HANDLER_HPP_
