import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import cv2
import gymnasium as gym
from gymnasium import spaces

class SoftObjectGraspEnv(gym.Env):
    """Environment for training a RL agent to grasp and lift a deformable object."""
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, xml_path="new_work2/shadow_hand/scene_right.xml", render_mode=None):
        """Initialize the environment with the given XML model file."""
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        
        # Setup camera for slip detection
        self.setup_camera()
        
        # Initialize observation and action spaces
        self.setup_spaces()
        
        # Initialize data storage
        self.force_data = {finger: [] for finger in ['FF', 'MF', 'RF', 'LF', 'TH']}
        self.deformation_data = []
        self.slip_detected = False
        self.prev_frame = None
        self.frames = []
        
        # Get site and geom IDs
        self.cyl_imu_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'cyl_imu')
        self.cyl_top_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'cyl_top')
        self.cylinder_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'soft_cylinder')
        self.cylinder_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'soft_cylinder')
        
        # Get cylinder properties
        self.initial_radius = self.model.geom_size[self.cylinder_geom_id, 0]
        self.initial_height = self.model.geom_size[self.cylinder_geom_id, 1] * 2
        
        # Define contact bodies mapping
        self.contact_bodies = {
            "rh_ffproximal": ('FF', 'J3'), "rh_ffmiddle": ('FF', 'J2'), "rh_ffdistal": ('FF', 'J1'),
            "rh_mfproximal": ('MF', 'J3'), "rh_mfmiddle": ('MF', 'J2'), "rh_mfdistal": ('MF', 'J1'),
            "rh_rfproximal": ('RF', 'J3'), "rh_rfmiddle": ('RF', 'J2'), "rh_rfdistal": ('RF', 'J1'),
            "rh_lfproximal": ('LF', 'J3'), "rh_lfmiddle": ('LF', 'J2'), "rh_lfdistal": ('LF', 'J1'),
            "rh_thmiddle": ('TH', 'J2'), "rh_thdistal": ('TH', 'J1')
        }
        
        # Set up motor control
        self.setup_motors()
        
        # Environment state variables
        self.phase = "grasp"  # Phases: "grasp", "lift"
        self.grasp_contacts = 0
        self.min_contacts_for_lift = 3
        self.lift_height = 0.0
        self.target_lift_height = 0.1
        self.initial_object_height = 0.0
        self.episode_steps = 0
        self.max_episode_steps = 500
        self.current_deformation = 0.0
        self.initial_volume = 0.0
        self.ff_contact_pos = None
        self.ff_contact_force = 0
        self.contact_points = []

        # For viewer
        self.viewer = None
    
    def setup_camera(self):
        """Set up camera for slip detection"""
        self.camera_name = "wrist_cam"
        self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if self.camera_id == -1:
            print(f"Warning: Camera '{self.camera_name}' not found, using default camera")
            self.camera_id = -1  # Use default camera
        self.renderer = mujoco.Renderer(self.model, height=224, width=224)
    
    def setup_spaces(self):
        """Set up observation and action spaces"""
        # 14 joint angles + 14 joint velocities + 5 fingertip forces + slip detection flag + deformation value
        obs_dim = 14 + 14 + 5 + 1 + 1
        
        # Define observation space
        # Low bounds
        obs_low = np.array(
            [-np.pi] * 14 +  # Joint angles
            [-10.0] * 14 +   # Joint velocities
            [0.0] * 5 +      # Fingertip forces
            [0.0] +          # Slip detection
            [0.0]            # Deformation
        )
        
        # High bounds
        obs_high = np.array(
            [np.pi] * 14 +   # Joint angles
            [10.0] * 14 +    # Joint velocities
            [50.0] * 5 +     # Fingertip forces
            [1.0] +          # Slip detection
            [0.1]            # Deformation
        )
        
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # 14 joint torques (13 finger joints + rail_to_base) as action space
        # Define reasonable torque limits for each joint
        action_low = np.array([0] * 14 + [-5000])  # Last action is rail_to_base
        action_high = np.array([0.3] * 14 + [0])   # Last action is rail_to_base
        
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        
        # Get joint IDs for the specified joints
        self.joint_names = [
            'rh_THJ1', 'rh_THJ2',
            'rh_FFJ1', 'rh_FFJ2', 'rh_FFJ3',
            'rh_MFJ1', 'rh_MFJ2', 'rh_MFJ3',
            'rh_RFJ1', 'rh_RFJ2', 'rh_RFJ3',
            'rh_LFJ1', 'rh_LFJ2', 'rh_LFJ3'
        ]
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                         for name in self.joint_names]
    
    def setup_motors(self):
        """Set up motors for control"""
        self.motor_names = [
            "rh_THJ1", "rh_THJ2",
            "rh_FFJ1", "rh_FFJ2", "rh_FFJ3",
            "rh_MFJ1", "rh_MFJ2", "rh_MFJ3",
            "rh_RFJ1", "rh_RFJ2", "rh_RFJ3",
            "rh_LFJ1", "rh_LFJ2", "rh_LFJ3",
            "rail_to_base"
        ]
        self.motor_ids = {name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) 
                          for name in self.motor_names}
    
    def detect_slip_from_camera(self, threshold=20, min_area=40):
        """Detect slip using camera feed and frame difference method"""
        # Render the current frame
        self.renderer.update_scene(self.data, camera=self.camera_id)
        current_frame = self.renderer.render().copy()
        self.frames.append(current_frame)
        
        # Convert to grayscale
        gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        
        if self.prev_frame is None:
            self.prev_frame = gray_frame
            return False
        
        # Compute frame difference
        frame_diff = cv2.absdiff(gray_frame, self.prev_frame)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Update previous frame
        self.prev_frame = gray_frame
        
        # Check for significant movement (slip)
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                # Draw contour on the frame for visualization if in render mode
                if self.render_mode == "human" and self.viewer is not None:
                    cv2.drawContours(current_frame, [contour], 0, (0, 0, 255), 2)
                    cv2.imshow("Slip Detection", current_frame)
                    cv2.waitKey(1)
                return True
        return False
    
    def calculate_signed_volume(self, point_a, point_b, point_c, origin):
        """
        Calculate the signed volume of a tetrahedron formed by three points and an origin
        Using the formula: Signed Volume = AO·(AB×AC)/6
        """
        ao = point_a - origin
        ab = point_b - point_a
        ac = point_c - point_a
        
        # Calculate the cross product AB × AC
        cross_product = np.cross(ab, ac)
        
        # Calculate the dot product AO·(AB×AC)
        dot_product = np.dot(ao, cross_product)
        
        # Return the signed volume (divided by 6)
        return dot_product / 6.0
    
    def calculate_volume_deformation(self):
        """Calculate volume deformation using signed volume method"""
        current_center_pos = self.data.site_xpos[self.cyl_imu_id]
        
        # Calculate deformation using signed volume method if we have enough contact points
        if len(self.contact_points) >= 3:
            total_volume = 0
            
            # Calculate volume using tetrahedra
            for i in range(len(self.contact_points) - 2):
                vol = self.calculate_signed_volume(
                    self.contact_points[i],
                    self.contact_points[i+1],
                    self.contact_points[i+2],
                    current_center_pos
                )
                total_volume += abs(vol)
            
            # Calculate deformation as change in volume
            volume_deformation = self.initial_volume - total_volume
            self.deformation_data.append(volume_deformation)
            self.current_deformation = volume_deformation
            return volume_deformation
        else:
            # Not enough contact points for volume calculation
            # Use vertical deformation as a fallback
            current_top_pos = self.data.site_xpos[self.cyl_top_id]
            current_height = np.linalg.norm(current_top_pos - current_center_pos)
            height_deformation = self.initial_height - current_height * 2
            
            if len(self.deformation_data) > 0:
                self.deformation_data.append(height_deformation)
            else:
                self.deformation_data.append(0)
            
            self.current_deformation = height_deformation
            return height_deformation
    
    def get_fingertip_forces(self):
        """Get forces at each fingertip"""
        fingertip_forces = np.zeros(5)
        
        # Map of fingertip body names to indices
        fingertip_map = {
            'rh_ffdistal': 0,  # FF
            'rh_mfdistal': 1,  # MF
            'rh_rfdistal': 2,  # RF
            'rh_lfdistal': 3,  # LF
            'rh_thdistal': 4   # TH
        }
        
        # Check all contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            # Get body IDs from geom IDs
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]
            
            # Get body names
            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
            
            # Check if contact involves a fingertip
            fingertip_idx = -1
            if body1_name in fingertip_map and body2_name == "soft_cylinder":
                fingertip_idx = fingertip_map[body1_name]
            elif body2_name in fingertip_map and body1_name == "soft_cylinder":
                fingertip_idx = fingertip_map[body2_name]
            
            if fingertip_idx >= 0:
                # Get contact force
                c_array = np.zeros(6, dtype=np.float64)
                mujoco._functions.mj_contactForce(self.model, self.data, i, c_array)
                contact_force = np.linalg.norm(c_array[:3])
                
                # Add to the appropriate fingertip force
                fingertip_forces[fingertip_idx] += contact_force
        
        return fingertip_forces
    
    def detect_contacts(self):
        """Detect contacts between fingers and cylinder"""
        # Reset contact data for this step
        self.ff_contact_pos = None
        self.ff_contact_force = 0
        self.contact_points = []
        self.grasp_contacts = 0
        
        # Check all contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            # Get body IDs from geom IDs
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]
            
            # Get body names
            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
            
            # Check if contact is between a hand/thumb body and the cylinder
            hand_body_name = None
            if body1_name in self.contact_bodies and body2_name == "soft_cylinder":
                hand_body_name = body1_name
                # Store contact point for deformation calculation
                self.contact_points.append(contact.pos.copy())
                self.grasp_contacts += 1
            elif body2_name in self.contact_bodies and body1_name == "soft_cylinder":
                hand_body_name = body2_name
                # Store contact point for deformation calculation
                self.contact_points.append(contact.pos.copy())
                self.grasp_contacts += 1
            
            if hand_body_name:
                finger_name, joint_type = self.contact_bodies[hand_body_name]
                
                # Get contact force
                c_array = np.zeros(6, dtype=np.float64)
                mujoco._functions.mj_contactForce(self.model, self.data, i, c_array)
                contact_force = np.linalg.norm(c_array[:3])
                
                # For FF specifically, record position for deformation
                if finger_name == 'FF' and joint_type == 'J1':
                    self.ff_contact_pos = contact.pos.copy()
                    self.ff_contact_force = contact_force
    
    def get_observations(self):
        """Get the current observation vector"""
        # Joint angles (14)
        joint_angles = np.zeros(14)
        for i, joint_id in enumerate(self.joint_ids):
            if joint_id != -1:
                joint_angles[i] = self.data.qpos[joint_id]
        
        # Joint velocities (14)
        joint_velocities = np.zeros(14)
        for i, joint_id in enumerate(self.joint_ids):
            if joint_id != -1:
                joint_velocities[i] = self.data.qvel[joint_id]
        
        # Fingertip forces (5)
        fingertip_forces = self.get_fingertip_forces()
        
        # Slip detection (1)
        slip_flag = np.array([float(self.slip_detected)])
        
        # Deformation (1)
        deformation = np.array([self.current_deformation])
        
        # Combine all observations
        obs = np.concatenate([
            joint_angles,
            joint_velocities,
            fingertip_forces,
            slip_flag,
            deformation
        ])
        
        return obs.astype(np.float32)
    
    def calculate_reward(self):
        """Calculate reward based on phase, contact, slip, and deformation"""
        # Base reward components
        fingertip_forces = self.get_fingertip_forces()
        contact_reward = sum(1 for force in fingertip_forces if force > 0.1)
        slip_penalty = -5 if self.slip_detected else 0
        deformation_penalty = -10 * self.current_deformation if self.current_deformation > 0.001 else 0
        
        # Phase-specific rewards
        if self.phase == "grasp":
            # Encourage making contact with the object
            reward = contact_reward + slip_penalty + deformation_penalty
            
            # Bonus for transitioning to lift phase
            if self.grasp_contacts >= self.min_contacts_for_lift:
                reward += 5
                
        elif self.phase == "lift":
            # Calculate lift height
            current_height = self.data.xpos[self.cylinder_body_id, 2]
            self.lift_height = current_height - self.initial_object_height
            
            # Reward for lifting height
            height_reward = 20 * (self.lift_height / self.target_lift_height)
            
            # Penalize slip more severely during lifting
            lift_slip_penalty = -10 if self.slip_detected else 0
            
            reward = contact_reward + lift_slip_penalty + deformation_penalty + height_reward
            
            # Bonus for successful lift
            if self.lift_height >= self.target_lift_height:
                reward += 50
        
        return reward
  
    def step(self, action):
        """Take a step in the environment with the given action"""
        self.episode_steps += 1
        
        # Apply joint torques from action
        for i, motor_name in enumerate(self.motor_names[:-1]):  # Exclude rail_to_base for now
            motor_id = self.motor_ids.get(motor_name, -1)
            if motor_id != -1 and i < len(action) - 1:  # Last action is for rail_to_base
                self.data.ctrl[motor_id] = action[i]
        
        # Handle rail_to_base separately based on phase
        rail_motor_id = self.motor_ids.get("rail_to_base", -1)
        if rail_motor_id != -1:
            if self.phase == "grasp":
                # During grasp phase, don't apply lift action
                self.data.ctrl[rail_motor_id] = 0
            else:  # lift phase
                # During lift phase, apply the lift action
                self.data.ctrl[rail_motor_id] = action[-1]  # Last action is rail_to_base
        
        # Step the simulation
        mujoco.mj_step(self.model, self.data)
        
        # Detect contacts for deformation calculation
        self.detect_contacts()
        
        # Detect slip using camera
        self.slip_detected = self.detect_slip_from_camera()
        
        # Calculate volume deformation
        self.calculate_volume_deformation()
        
        # Check if we should transition from grasp to lift phase
        if self.phase == "grasp" and self.grasp_contacts >= self.min_contacts_for_lift:
            self.phase = "lift"
            self.initial_object_height = self.data.xpos[self.cylinder_body_id, 2]
            print("Transitioning to lift phase")
        
        # Get observations
        obs = self.get_observations()
        
        # Calculate reward
        reward = self.calculate_reward()
        
        # Check if episode is done
        # Done if object is dropped, excessive deformation, successful lift, or max steps reached
        cylinder_height = self.data.xpos[self.cylinder_body_id, 2]
        dropped = cylinder_height < 0.1 and self.phase == "lift"
        excessive_deformation = self.current_deformation > 0.1 * self.initial_volume
        successful_lift = self.phase == "lift" and self.lift_height >= self.target_lift_height
        timeout = self.episode_steps >= self.max_episode_steps
        
        terminated = dropped or excessive_deformation or successful_lift
        truncated = timeout
        
        # Info dictionary
        info = {
            'phase': self.phase,
            'grasp_contacts': self.grasp_contacts,
            'deformation': self.current_deformation,
            'slip_detected': self.slip_detected,
            'fingertip_forces': self.get_fingertip_forces(),
            'dropped': dropped,
            'excessive_deformation': excessive_deformation,
            'successful_lift': successful_lift,
            'lift_height': self.lift_height if self.phase == "lift" else 0.0
        }
        
        # Render if in human mode
        if self.render_mode == "human" and self.viewer is not None:
            self.render()
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial thumb position
        joint_id_thj4 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rh_THJ4")
        if joint_id_thj4 != -1:
            self.data.qpos[joint_id_thj4] = 1.22173
      
        # Reset camera frame
        self.prev_frame = None
        self.frames = []
        
        # Reset contact data
        self.ff_contact_pos = None
        self.ff_contact_force = 0
        self.contact_points = []
        self.slip_detected = False
        self.deformation_data = []
        self.current_deformation = 0.0
        
        # Reset phase
        self.phase = "grasp"
        self.grasp_contacts = 0
        self.lift_height = 0.0
        
        # Reset episode variables
        self.episode_steps = 0
        
        # Step once to get initial state
        mujoco.mj_step(self.model, self.data)
        
        # Calculate initial volume
        self.initial_volume = np.pi * (self.initial_radius**2) * self.initial_height
        self.initial_object_height = self.data.xpos[self.cylinder_body_id, 2]
        
        # Return initial observation
        return self.get_observations(), {}
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            
            # Sync the viewer
            if self.viewer.is_running():
                self.viewer.sync()
                
                # Also render camera view if available
                if len(self.frames) > 0:
                    cv2.imshow("Camera View", self.frames[-1])
                    cv2.waitKey(1)
    
    def close(self):
        """Close the environment"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        
        cv2.destroyAllWindows()
