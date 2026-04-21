import time
from typing import Optional

import rclpy
from example_interfaces.srv import SetBool
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    OrientationConstraint,
    PositionConstraint,
    RobotState,
    RobotTrajectory,
)
from moveit_msgs.srv import GetMotionPlan
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

# from abb_egm_interfaces.action import ExecuteTrajectory #HERE

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from cv2 import aruco
import matplotlib.pyplot as plt

class ImageCapture:
    # SHARED_DIR = Path("/realsense_shared")
    SHARED_DIR = Path(r"C:\Users\alexl\Documents\Python_Scripts\ARC380\ARC380_Team_1\arc380_s26_team_1\realsense_shared")
    REQUEST_PATH = SHARED_DIR / "request.json"
    READY_PATH = SHARED_DIR / "ready.json"
    COLOR_PATH = SHARED_DIR / "color.png"
    DEPTH_PATH = SHARED_DIR / "depth.npy"
    META_PATH = SHARED_DIR / "meta.json"

    POLL_INTERVAL_SEC = 0.1
    TIMEOUT_SEC = 10.0

    aruco_corners: dict[int, np.ndarray] = {
    0: np.array([
        [-0.068,0.271,0.021],
        [-0.094,0.271,0.021],
        [-0.094,0.297,0.021],
        [-0.068,0.297,0.021],
    ]),
    1: np.array([
        [-0.233,0.271,0.021],
        [-0.259,0.271,0.021],
        [-0.259,0.297,0.021],
        [-0.233,0.297,0.021],
    ]),
    2: np.array([
        [-0.233,0.500,0.021],
        [-0.259,0.500,0.021],
        [-0.259,0.525,0.021],
        [-0.233,0.525,0.021],
    ]),
    3: np.array([
        [-0.068,0.500,0.021],
        [-0.094,0.500,0.021],
        [-0.094,0.525,0.021],
        [-0.068,0.525,0.021],
    ]),
}


    @staticmethod
    def read_json(path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
        tmp_path.replace(path)

    @staticmethod
    def get_next_request_id() -> int:
        if ImageCapture.READY_PATH.exists():
            try:
                ready = ImageCapture.read_json(ImageCapture.READY_PATH)
                if isinstance(ready.get("request_id"), int):
                    return ready["request_id"] + 1
            except Exception:
                pass

        if ImageCapture.REQUEST_PATH.exists():
            try:
                req = ImageCapture.read_json(ImageCapture.REQUEST_PATH)
                if isinstance(req.get("request_id"), int):
                    return req["request_id"] + 1
            except Exception:
                pass

        return 1

    @staticmethod
    def request_capture(timeout_sec: float = TIMEOUT_SEC) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        ImageCapture.SHARED_DIR.mkdir(parents=True, exist_ok=True)

        request_id = ImageCapture.get_next_request_id()

        if ImageCapture.READY_PATH.exists():
            ImageCapture.READY_PATH.unlink()

        request_payload = {
            "request_id": request_id,
            "capture": True,
        }
        ImageCapture.atomic_write_json(ImageCapture.REQUEST_PATH, request_payload)

        deadline = time.monotonic() + timeout_sec

        while time.monotonic() < deadline:
            if ImageCapture.READY_PATH.exists():
                try:
                    ready = ImageCapture.read_json(ImageCapture.READY_PATH)
                except Exception:
                    time.sleep(ImageCapture.POLL_INTERVAL_SEC)
                    continue

                if ready.get("request_id") != request_id:
                    time.sleep(ImageCapture.POLL_INTERVAL_SEC)
                    continue

                status = ready.get("status")
                if status != "ok":
                    raise RuntimeError(f"Capture failed: {ready}")

                if not ImageCapture.COLOR_PATH.exists():
                    raise FileNotFoundError(f"Missing file: {ImageCapture.COLOR_PATH}")
                if not ImageCapture.DEPTH_PATH.exists():
                    raise FileNotFoundError(f"Missing file: {ImageCapture.DEPTH_PATH}")
                if not ImageCapture.META_PATH.exists():
                    raise FileNotFoundError(f"Missing file: {ImageCapture.META_PATH}")

                color = cv2.imread(str(ImageCapture.COLOR_PATH), cv2.IMREAD_COLOR)
                if color is None:
                    raise RuntimeError(f"Failed to load color image from {ImageCapture.COLOR_PATH}")

                depth = np.load(str(ImageCapture.DEPTH_PATH))
                meta = ImageCapture.read_json(ImageCapture.META_PATH)

                return color, depth, meta

            time.sleep(ImageCapture.POLL_INTERVAL_SEC)

        raise TimeoutError(f"Timed out waiting for capture response after {timeout_sec} seconds")
    
    @staticmethod
    def removePerspective(rgbImg):
        # Load the predefined dictionary where our markers are printed from
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

        # Load the default detector parameters
        detector_params = aruco.DetectorParameters()

        # Create an ArucoDetector using the dictionary and detector parameters
        detector = aruco.ArucoDetector(dictionary, detector_params)

        corners, ids, rejected = detector.detectMarkers(rgbImg)

        # Sort corners based on id
        ids = ids.flatten()
        #print(ids)

        # Sort the corners based on the ids
        corners = np.array([corners[i] for i in np.argsort(ids)])
        # print(corners.shape)

        # Remove dimensions of size 1
        corners = np.squeeze(corners)
        # print(corners)

        # Sort the ids
        ids = np.sort(ids)

        # Extract source points corresponding to the exterior bounding box corners of the 4 markers
        src_pts = np.array([corners[3][3], corners[0][0], corners[1][1], corners[2][2]], dtype='float32')
        # print(src_pts)

        # width = 10      # inches
        # height = 7.5    # inches old
        width = 24.05512      # inches
        height = 32.83465    # inches ours
        ppi = int(min(1080/width, 1920/height))       # pixels per inch (standard resolution for most screens - can be any arbitrary value that still preserves information)
        dst_pts = np.array([[0, 0], [0, width*ppi], [height*ppi, width*ppi], [height*ppi, 0]], dtype='float32')
        # print(dst_pts)

        # Compute the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # print(M)

        # Apply the perspective transformation to the input image
        # print(rgbImg.shape[1])
        corrected_img = cv2.warpPerspective(rgbImg, M, (rgbImg.shape[1], rgbImg.shape[0]))

        # Crop the output image to the specified dimensions
        corrected_img = corrected_img[:int(width*ppi), :int(height*ppi)]

        # plt.imshow(cv2.cvtColor(corrected_img, cv2.COLOR_BGR2RGB))
        # plt.title('Perspective corrected image')
        # plt.gca().invert_xaxis()
        # plt.show()

        return corrected_img

    @staticmethod
    def getClusterCords(flatImg):
        # Run k-means clustering on the image

        # Reshape our image data to a flattened list of RGB values
        img_data = flatImg.reshape((-1, 3))
        img_data = np.float32(img_data)

        # Define the number of clusters
        k = 4 #black codes, white background, brown blocks, blue background

        # Define the criteria for the k-means algorithm
        # This is a tuple with three elements: (type of termination criteria, maximum number of iterations, epsilon/required accuracy)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # Run the k-means algorithm
        # Parameters: data, number of clusters, best labels, criteria, number of attempts, initial centers
        _, labels, centers = cv2.kmeans(img_data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # The output of the k-means algorithm gives the centers as floating point values
        # We need to convert these back to uint8 to be able to use them as pixel values
        centers = np.uint8(centers)

        # Rebuild the image using the labels and centers
        kmeans_data = centers[labels.flatten()]
        kmeans_img = kmeans_data.reshape(flatImg.shape)
        labels = labels.reshape(flatImg.shape[:2])

        # plt.imshow(cv2.cvtColor(kmeans_img, cv2.COLOR_BGR2RGB))
        # plt.title(f'Image classification using k-means clustering (k = {k})')
        # plt.gca().invert_yaxis()
        # plt.show()

        # Identify the cluster that is closest to brown color
        block_brown = np.array([123, 159, 186])
        distances = np.linalg.norm(centers - block_brown, axis=1)
        block_cluster_label = np.argmin(distances)

        # Create a mask image for this label
        # All pixels that belong to this cluster will be white, and all others will be black
        mask_img = np.zeros(kmeans_img.shape[:2], dtype='uint8')
        mask_img[labels == block_cluster_label] = 255

        # plt.imshow(mask_img, cmap='gray')
        # plt.title(f'Mask image for cluster {block_cluster_label} corresponding to dark green')
        # plt.gca().invert_yaxis()
        # plt.show()

        # Segment continuous regions
        # Parameters: input image, contour retrieval mode, contour approximation method
        contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Visualize the contours
        # Parameters for drawContours: input image, contours, contour index (-1 means all contours), color, thickness
        contour_img = flatImg.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)

        # Get area of each region
        areas = [cv2.contourArea(contour) for contour in contours]
        # print(f'Area of each region: {areas}')

        # Calculate the expected pixel area
        # width = 10      # inches
        # height = 7.5    # inches old
        width = 24.05512      # inches
        height = 32.83465    # inches ours
        ppi_2 = int(min(1080/width, 1920/height))
        expected_area = (1.9685 * 0.905512) * (ppi_2**2)
        area_tolerance = expected_area * 0.2
        # print(f'expected_area: {expected_area}')

        # Find the contour with the closest area to the expected area
        sorted_area_diff = np.sort(np.abs(np.array(areas) - expected_area))
        block_indices = np.where(np.abs(np.array(areas) - expected_area) < area_tolerance)[0]

        # print(block_indices)

        u_c = np.zeros(len(block_indices))
        v_c = np.zeros(len(block_indices))
        angle = np.zeros(len(block_indices))
        for index, i in enumerate(block_indices):
            selected_contour = contours[block_indices[index]]
            # x, y, w, h = cv2.boundingRect(selected_contour)
            # u_c[index] = x + w//2
            # v_c[index] = y + h//2

            moments = cv2.moments(selected_contour)
            u_c[index] = round(moments['m10']/moments['m00'])
            v_c[index] = round(moments['m01']/moments['m00'])

            rect = cv2.minAreaRect(selected_contour) # minAreaRect returns a Box2D structure. A Box2D structure is a tuple of ((x, y), (w, h), angle).
            if(rect[1][0] > rect[1][1]):
                angle[index] = ((rect[2] + 90) % 180) - 90
            else:
                angle[index] = ((rect[2] + 180) % 180) - 90
            # print(f'x,y: {rect[0]}, w,h: {rect[1]}')


        for i in range (len(block_indices)):
            print(f'x: {u_c[i]}, y: {v_c[i]}, angle: {angle[i]}')

        # Draw the center of the selected contour
        center_img = flatImg.copy()
        for i in range(len(u_c)):    
            cv2.circle(center_img, (int(u_c[i]), int(v_c[i])), 5, (255, 255, 0), -1)

        
        u_c_m = u_c / ppi_2 * (25.4 / 1000)
        v_c_m = v_c / ppi_2 * (25.4 / 1000)
        print(f'u_c_m: {u_c_m}')
        print(f'v_c_m: {v_c_m}')

        # HERE - if the tags are angled, this needs to be modified
        # aruco_origin_x = ImageCapture.aruco_corners[3][0][0]
        # aruco_origin_y = ImageCapture.aruco_corners[3][0][1]
        aruco_origin_x = 0 # GET MEASUREMENTS FOR THESE
        aruco_origin_y = 0 # GET MEASUREMENTS FOR THESE
        aruco_opp_x = height * (25.4 / 1000) # GET MEASUREMENTS FOR THESE
        aruco_opp_y = width * (25.4 / 1000) # GET MEASUREMENTS FOR THESE
        real_corner_angle = np.arctan2(aruco_opp_y - aruco_origin_y, aruco_opp_x - aruco_origin_x)
        ideal_corner_angle = np.arctan2(width, height)

        corner_angle_diff = real_corner_angle - ideal_corner_angle
        print(corner_angle_diff)

        work_x = aruco_origin_x + (u_c_m * np.cos(corner_angle_diff)) - (v_c_m * np.sin(corner_angle_diff))
        work_y = aruco_origin_y + (v_c_m * np.cos(corner_angle_diff)) + (u_c_m * np.sin(corner_angle_diff))
        # angle = angle - 90
        print(f'work_x: {work_x}')
        print(f'work_y: {work_y}')
        
        plt.imshow(cv2.cvtColor(center_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Center of the selected contour for label {block_cluster_label}')
        plt.gca().invert_yaxis()
        plt.show()

        return work_x, work_y, angle




class EGMClient(Node):
    def __init__(self):
        super().__init__("egm_client")
        self.set_parameters([Parameter("use_sim_time", value=True)])

        # MoveIt planning service
        self.plan_cli = self.create_client(GetMotionPlan, "/plan_kinematic_path")
        while not self.plan_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for /plan_kinematic_path service...")
        self.get_logger().info("/plan_kinematic_path service available.")

        # EGM execution action
        self.execute_traj_ac = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")

        self.gripper_client = self.create_client(SetBool, "/egm_controller/control/set_gripper")

    @staticmethod
    def _make_start_state(joint_names, joint_positions) -> RobotState:
        rs = RobotState()
        js = JointState()
        js.name = list(joint_names)
        js.position = list(joint_positions)
        rs.joint_state = js
        return rs

    @staticmethod
    def _make_position_constraint(
        link_name: str,
        frame_id: str,
        target_xyz,
        tolerance_xyz=(0.005, 0.005, 0.005),
        weight: float = 1.0,
    ) -> PositionConstraint:
        pc = PositionConstraint()
        pc.header.frame_id = frame_id
        pc.link_name = link_name
        pc.weight = float(weight)

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [
            float(2.0 * tolerance_xyz[0]),
            float(2.0 * tolerance_xyz[1]),
            float(2.0 * tolerance_xyz[2]),
        ]

        bv = BoundingVolume()
        bv.primitives = [box]

        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x = float(target_xyz[0])
        pose.pose.position.y = float(target_xyz[1])
        pose.pose.position.z = float(target_xyz[2])
        pose.pose.orientation.w = 1.0
        bv.primitive_poses = [pose.pose]

        pc.constraint_region = bv
        return pc

    @staticmethod
    def _make_orientation_constraint(
        link_name: str,
        frame_id: str,
        target_quat_wxyz,
        tolerance_rpy=(0.05, 0.05, 0.05),
        weight: float = 1.0,
    ) -> OrientationConstraint:
        oc = OrientationConstraint()
        oc.header.frame_id = frame_id
        oc.link_name = link_name
        oc.weight = float(weight)

        oc.orientation.w = float(target_quat_wxyz[0])
        oc.orientation.x = float(target_quat_wxyz[1])
        oc.orientation.y = float(target_quat_wxyz[2])
        oc.orientation.z = float(target_quat_wxyz[3])

        oc.absolute_x_axis_tolerance = float(tolerance_rpy[0])
        oc.absolute_y_axis_tolerance = float(tolerance_rpy[1])
        oc.absolute_z_axis_tolerance = float(tolerance_rpy[2])
        return oc

    @staticmethod
    def _make_joint_constraint(
        joint_name: str,
        position: float,
        tolerance_above: float = 1e-3,
        tolerance_below: float = 1e-3,
        weight: float = 1.0,
    ) -> JointConstraint:
        jc = JointConstraint()
        jc.joint_name = joint_name
        jc.position = float(position)
        jc.tolerance_above = float(tolerance_above)
        jc.tolerance_below = float(tolerance_below)
        jc.weight = float(weight)
        return jc

    def _call_motion_plan(self, mpr: MotionPlanRequest) -> Optional[RobotTrajectory]:
        req = GetMotionPlan.Request()
        req.motion_plan_request = mpr

        future = self.plan_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error("Motion planning service call failed.")
            return None

        resp = future.result()
        mres = resp.motion_plan_response

        if mres.error_code.val != mres.error_code.SUCCESS:
            self.get_logger().error(f"Planning failed. MoveItErrorCodes.val = {mres.error_code.val}")
            return None

        traj = mres.trajectory
        jt = traj.joint_trajectory
        self.get_logger().info(
            f"Planning succeeded. JointTrajectory has {len(jt.points)} points for joints: {list(jt.joint_names)}"
        )
        if jt.points:
            last = jt.points[-1]
            self.get_logger().info("Last configuration:")
            for n, p in zip(jt.joint_names, last.positions):
                self.get_logger().info(f"  {n}: {p:.6f}")
        return traj

    def plan_arm_to_pose_constraints(
        self,
        group_name: str,
        link_name: str,
        frame_id: str,
        goal_xyz: tuple[float, float, float],
        start_joint_names: list[str] | None = None,
        start_joint_positions: list[float] | None = None,
        goal_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        pos_tolerance_xyz: tuple[float, float, float] = (0.001, 0.001, 0.001),
        ori_tolerance_rpy: tuple[float, float, float] = (0.001, 0.001, 0.001),
        allowed_planning_time: float = 5.0,
        num_attempts: int = 5,
        max_velocity_scaling: float = 0.2,
        max_acceleration_scaling: float = 0.2,
        planner_id: str = "",
    ) -> Optional[RobotTrajectory]:
        mpr = MotionPlanRequest()
        mpr.group_name = group_name
        if planner_id:
            mpr.planner_id = planner_id

        if start_joint_names is not None and start_joint_positions is not None:
            mpr.start_state = self._make_start_state(start_joint_names, start_joint_positions)

        constraints = Constraints()
        constraints.position_constraints = [
            self._make_position_constraint(
                link_name=link_name,
                frame_id=frame_id,
                target_xyz=goal_xyz,
                tolerance_xyz=pos_tolerance_xyz,
            )
        ]
        constraints.orientation_constraints = [
            self._make_orientation_constraint(
                link_name=link_name,
                frame_id=frame_id,
                target_quat_wxyz=goal_quat_wxyz,
                tolerance_rpy=ori_tolerance_rpy,
            )
        ]
        mpr.goal_constraints = [constraints]

        mpr.allowed_planning_time = float(allowed_planning_time)
        mpr.num_planning_attempts = int(num_attempts)
        mpr.max_velocity_scaling_factor = float(max_velocity_scaling)
        mpr.max_acceleration_scaling_factor = float(max_acceleration_scaling)

        return self._call_motion_plan(mpr)

    def plan_gripper_to_joint_positions(
        self,
        group_name: str,
        goal_joint_names: list[str],
        goal_joint_positions: list[float],
        start_joint_names: list[str] | None = None,
        start_joint_positions: list[float] | None = None,
        tolerance: float = 1e-3,
        allowed_planning_time: float = 2.0,
        num_attempts: int = 3,
        max_velocity_scaling: float = 1.0,
        max_acceleration_scaling: float = 1.0,
        planner_id: str = "",
    ) -> Optional[RobotTrajectory]:
        """
        Plans a joint-space motion for the gripper group.
        For a non-mimic 2-finger gripper, pass both finger joints.
        Example:
          goal_joint_names = ["left_finger_joint", "right_finger_joint"]
          goal_joint_positions = [0.01, 0.01]
        """
        if len(goal_joint_names) != len(goal_joint_positions):
            self.get_logger().error("goal_joint_names and goal_joint_positions must match.")
            return None

        mpr = MotionPlanRequest()
        mpr.group_name = group_name
        if planner_id:
            mpr.planner_id = planner_id

        if start_joint_names is not None and start_joint_positions is not None:
            mpr.start_state = self._make_start_state(start_joint_names, start_joint_positions)

        constraints = Constraints()
        constraints.joint_constraints = [
            self._make_joint_constraint(
                joint_name=n,
                position=p,
                tolerance_above=tolerance,
                tolerance_below=tolerance,
            )
            for n, p in zip(goal_joint_names, goal_joint_positions)
        ]
        mpr.goal_constraints = [constraints]

        mpr.allowed_planning_time = float(allowed_planning_time)
        mpr.num_planning_attempts = int(num_attempts)
        mpr.max_velocity_scaling_factor = float(max_velocity_scaling)
        mpr.max_acceleration_scaling_factor = float(max_acceleration_scaling)

        return self._call_motion_plan(mpr)

    def execute_moveit_trajectory(self, traj: RobotTrajectory) -> bool:

        if not self.execute_traj_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("/execute_trajectory action not available.")
            return False

        goal = ExecuteTrajectory.Goal()
        goal.trajectory = traj.joint_trajectory
        goal.stop_active_motion = True

        send_future = self.execute_traj_ac.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("EGM execution goal rejected.")
            return False

        self.get_logger().info("EGM execution goal accepted.")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()

        if result is None:
            self.get_logger().error("Failed to get EGM execution result.")
            return False

        if not result.result.success:
            self.get_logger().error(f"EGM execution failed. Message: {result.result.message}")
            return False

        self.get_logger().info("EGM execution succeeded.")
        return True

    def send_gripper_command(
        self,
        position: float,
        max_velocity: float = 0.02,
        max_effort: float = 0.0,
        joint_name: str = "left_finger_joint",
    ) -> bool:
        if not self.gripper_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Gripper control service not available.")
            return False

        req = SetBool.Request()
        req.data = position > 1e-6
        future = self.gripper_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error("Failed to call gripper control service.")
            return False

        response = future.result()
        if not response.success:
            self.get_logger().error(f"Gripper control failed: {response.message}")
            return False
        self.get_logger().info(f"Gripper control succeeded: {response.message}")
        time.sleep(2.0)
        return True

class Helpers:
    @staticmethod
    def euler_angles_to_quarternion(euler_angles: np.ndarray) -> np.ndarray:
        """
        Convert Euler angles to a 3x3 rotation matrix using an intrinsic z-y'-x" convention.

        Parameters
        ----------
        euler_angles : np.ndarray
            Array of shape (3,) containing [yaw, pitch, roll] in degrees.

        Returns
        -------
        np.ndarray
            3x3 rotation matrix.

        """
        matrix = None

        # ================================== YOUR CODE HERE ==================================

        yaw = np.deg2rad(euler_angles[0])
        pitch = np.deg2rad(euler_angles[1])
        roll = np.deg2rad(euler_angles[2])
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])

        matrix = R_z @ R_y @ R_x

        # ====================================================================================

        """
        Convert a 3x3 rotation matrix to a unit quaternion.

        Parameters
        ----------
        matrix : np.ndarray
            3x3 rotation matrix.

        Returns
        -------
        np.ndarray
            Array of shape (4,) containing [w, x, y, z] where w is the scalar part.

        """
        quaternion = None

        # ================================== YOUR CODE HERE ==================================

        M00 = np.trace(matrix)
        M11 = matrix[0, 0]
        M22 = matrix[1, 1]
        M33 = matrix[2, 2]
        M = np.array([M00, M11, M22, M33])

        p = np.zeros(4)

        i = np.argmax(M)
        if i == 0:
            p[0] = np.sqrt(1 + M00)
            p[1] = (matrix[2, 1] - matrix[1, 2]) / p[0]
            p[2] = (matrix[0, 2] - matrix[2, 0]) / p[0]
            p[3] = (matrix[1, 0] - matrix[0, 1]) / p[0]
        elif i == 1:
            p[1] = np.sqrt(1 + 2*M11 - M00)
            p[0] = (matrix[2, 1] - matrix[1, 2]) / p[1]
            p[2] = (matrix[1, 0] + matrix[0, 1]) / p[1]
            p[3] = (matrix[0, 2] + matrix[2, 0]) / p[1]
        elif i == 2:
            p[2] = np.sqrt(1 + 2*M22 - M00)
            p[0] = (matrix[0, 2] - matrix[2, 0]) / p[2]
            p[1] = (matrix[1, 0] + matrix[0, 1]) / p[2]
            p[3] = (matrix[2, 1] + matrix[1, 2]) / p[2]
        else:
            p[3] = np.sqrt(1 + 2*M33 - M00)
            p[0] = (matrix[1, 0] - matrix[0, 1]) / p[3]
            p[1] = (matrix[0, 2] + matrix[2, 0]) / p[3]
            p[2] = (matrix[2, 1] + matrix[1, 2]) / p[3]

        quaternion = 0.5 * p
        if(quaternion[0] < 0):
            quaternion = -quaternion

        # ====================================================================================

        return quaternion


def main():
    # rclpy.init()

    # node = EGMClient()

    # gripper_open = 0.00
    # gripper_closed = 0.01

    # node.send_gripper_command(
    #     position=gripper_open,
    #     max_velocity=0.05,
    # )


    # color, depth, meta = ImageCapture.request_capture()

    img_path = Path(r"C:\Users\alexl\Documents\Python_Scripts\ARC380\ARC380_Team_1\arc380_s26_team_1\realsense_shared\color.png")
    color = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    

    flatImg = ImageCapture.removePerspective(color)

    x_blocks, y_blocks, angle_blocks = ImageCapture.getClusterCords(flatImg)

    return

    ###########################################

    a = 0.70717
    b = 0.92388
    c = 0.38268

    tower_x = 0.419
    tower_y = 0.221
    parallel_dx = 0.043
    parallel_dy = 0.043
    diagonal_dx = 0.043 * 0.707
    diagonal_dy = 0.043 * 0.707
    # base_z = 0.014 / 2
    base_z = 0
    dz = 0.014
    large_clearance_z = 0.1
    drop_clearance_z = 0.001 #was 0.01 -> 0.004 -> 0.00

    # first_block_x = 0.0
    # first_block_y = 0.480
    above_block_z = 0.1
    around_block_z = 0.025 #was 0.032 -> 0.025
    # holder_dx = 0.06
    # holder_dy = 0.06
    # num_y_blocks = 5


    tower_block_points = [
        [[tower_x+parallel_dx, tower_y, base_z], [0.0, a, a, 0.0]],
        [[tower_x, tower_y+parallel_dy, base_z], [0.0, 1.0, 0.0, 0.0]],
        [[tower_x-parallel_dx, tower_y, base_z], [0.0, a, a, 0.0]],
        [[tower_x, tower_y-parallel_dy, base_z], [0.0, 1.0, 0.0, 0.0]],
        [[tower_x+diagonal_dx, tower_y+diagonal_dy, base_z+dz], [0.0, b, -c, 0.0]],
        [[tower_x-diagonal_dx, tower_y+diagonal_dy, base_z+dz], [0.0, b, c, 0.0]],
        [[tower_x-diagonal_dx, tower_y-diagonal_dy, base_z+dz], [0.0, b, -c, 0.0]],
        [[tower_x+diagonal_dx, tower_y-diagonal_dy, base_z+dz], [0.0, b, c, 0.0]],
    ]

    # for x, y, angle in zip(x_all, y_all, angle_all):


    index = 0
    for point, angle in tower_block_points:

        #Open gripper
        node.send_gripper_command(
            position=gripper_open,
            max_velocity=0.05,
        )

        #Move to above block
        quart = Helpers.euler_angles_to_quarternion([angle_blocks[index], 0, 180])
        arm_traj = node.plan_arm_to_pose_constraints(
            group_name="arm",
            link_name="gripper_tcp_calibrated",
            frame_id="world",
            goal_xyz=(x_blocks[index], y_blocks[index], above_block_z),
            goal_quat_wxyz=quart,
        )
        if arm_traj is not None:
            node.execute_moveit_trajectory(arm_traj)

        #Move down
        arm_traj = node.plan_arm_to_pose_constraints(
            group_name="arm",
            link_name="gripper_tcp_calibrated",
            frame_id="world",
            goal_xyz=(x_blocks[index], y_blocks[index], around_block_z),
            goal_quat_wxyz=quart,
        )
        if arm_traj is not None:
            node.execute_moveit_trajectory(arm_traj)

        #Grab block
        node.send_gripper_command(
            position=gripper_closed,
            max_velocity=0.05,
        )

        #Move up
        arm_traj = node.plan_arm_to_pose_constraints(
            group_name="arm",
            link_name="gripper_tcp_calibrated",
            frame_id="world",
            goal_xyz=(x_blocks[index], y_blocks[index], above_block_z),
            goal_quat_wxyz=quart,
        )
        if arm_traj is not None:
            node.execute_moveit_trajectory(arm_traj)

        #Move to above block's placement point
        arm_traj = node.plan_arm_to_pose_constraints(
            group_name="arm",
            link_name="gripper_tcp_calibrated",
            frame_id="world",
            goal_xyz=(point[0], point[1], point[2] + around_block_z + large_clearance_z),
            goal_quat_wxyz=(angle[0], angle[1], angle[2], angle[3]),
        )
        if arm_traj is not None:
            node.execute_moveit_trajectory(arm_traj)

        #Move down
        arm_traj = node.plan_arm_to_pose_constraints(
            group_name="arm",
            link_name="gripper_tcp_calibrated",
            frame_id="world",
            goal_xyz=(point[0], point[1], point[2] + around_block_z + drop_clearance_z),
            goal_quat_wxyz=(angle[0], angle[1], angle[2], angle[3]),
        )
        if arm_traj is not None:
            node.execute_moveit_trajectory(arm_traj)

        #Drop block
        node.send_gripper_command(
            position=gripper_open,
            max_velocity=0.05,
        )

        #Move Up
        arm_traj = node.plan_arm_to_pose_constraints(
            group_name="arm",
            link_name="gripper_tcp_calibrated",
            frame_id="world",
            goal_xyz=(point[0], point[1], point[2] + around_block_z + large_clearance_z),
            goal_quat_wxyz=(angle[0], angle[1], angle[2], angle[3]),
        )
        if arm_traj is not None:
            node.execute_moveit_trajectory(arm_traj)

        index += 1




    ########################################

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
