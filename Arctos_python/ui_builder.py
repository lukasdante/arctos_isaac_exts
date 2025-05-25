# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os
import math
import numpy as np
import torch
from pathlib import Path
import omni.timeline
import omni.ui as ui
import omni.usd
from pxr import Usd, Sdf, UsdGeom
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.prims import get_prim_object_type
from isaacsim.core.utils.types import ArticulationAction, ArticulationActions
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils import extensions
from isaacsim.gui.components.element_wrappers import (
    CollapsableFrame, DropDown, FloatField, TextBlock, StateButton, Button, StringField, XYPlot)
from isaacsim.gui.components.ui_utils import get_style
from isaacsim.storage.native.nucleus import get_assets_root_path
import omni.graph.core as og
from .model import GaussianPolicy
import torch

class UIBuilder:
    def __init__(self):
        # Frames are sub-windows that can contain multiple UI elements
        self.frames = []

        # UI elements created using a UIElementWrapper from isaacsim.gui.components.element_wrappers
        self.wrapped_ui_elements = []

        # Get access to the timeline to control stop/pause/play programmatically
        self._timeline = omni.timeline.get_timeline_interface()



        # Run initialization for the provided example
        self._on_init()

    ###################################################################################
    #           The Functions Below Are Called Automatically By extension.py
    ###################################################################################

    def on_menu_callback(self):
        """Callback for when the UI is opened from the toolbar.
        This is called directly after build_ui().
        """
        # Reset internal state when UI window is closed and reopened
        self._invalidate_articulation()

        self._selection_menu.repopulate()

        # Handles the case where the user loads their Articulation and
        # presses play before opening this extension
        if self._timeline.is_playing():
            self._stop_text.visible = False
            self._enable_ROS_2_button.visible = False
        elif self._timeline.is_stopped():
            self._stop_text.visible = True
            self._enable_ROS_2_button.visible = False

    def on_timeline_event(self, event):
        """Callback for Timeline events (Play, Pause, Stop)

        Args:
            event (omni.timeline.TimelineEventType): Event Type
        """
        pass

    def on_physics_step(self, step):
        """Callback for Physics Step.
        Physics steps only occur when the timeline is playing

        Args:
            step (float): Size of physics step
        """
        if not self._reaching:
            return
        
        # 1) collect state, get actions, apply them…
        state = self.collect_state()
        with torch.no_grad():
            actions = self.model(state)[0]
        # apply to joints…
        self.last_actions = actions

        # Actuate the simulation and the motors
        
        for i in range(3):
            field = self._joint_position_float_fields[i]

            # The policy outputs relative joint positions, so add the current joint position to it
            position = actions[i].item()
            field.set_value(position)

            robot_action = ArticulationAction(
                joint_positions=np.array([position]),
                joint_velocities=np.array([0]),
                joint_indices=np.array([i]),
            )
        
            self.articulation.apply_action(robot_action)
        
        # 2) compute error
        err = torch.norm(
            torch.tensor(self.get_prim_transformation("/World/arctos/right_jaw")[0]) -
            torch.tensor(self.get_prim_transformation("/World/Box")[0]),
            p=2
        )

        if err <= self._target_tolerance:
            self._reaching = False
        pass

    def on_stage_event(self, event):
        """Callback for Stage Events

        Args:
            event (omni.usd.StageEventType): Event Type
        """
        if event.type == int(omni.usd.StageEventType.ASSETS_LOADED):  # Any asset added or removed
            self._selection_menu.repopulate()
        elif event.type == int(omni.usd.StageEventType.SIMULATION_START_PLAY):  # Timeline played
            # Treat a playing timeline as a trigger for selecting an Articulation
            self._selection_menu.trigger_on_selection_fn_with_current_selection()
            self._stop_text.visible = False
            self._enable_ROS_2_button.visible = False
        elif event.type == int(omni.usd.StageEventType.SIMULATION_STOP_PLAY):  # Timeline stopped
            # Ignore pause events
            if self._timeline.is_stopped():
                self._invalidate_articulation()
                self._stop_text.visible = True
                self._enable_ROS_2_button.visible = False

    def cleanup(self):
        """
        Called when the stage is closed or the extension is hot reloaded.
        Perform any necessary cleanup such as removing active callback functions
        Buttons imported from isaacsim.gui.components.element_wrappers implement a cleanup function that should be called
        """
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()
        
    def prepare_stage(self):
        stage = omni.usd.get_context().get_stage()
        if stage:
            self.stage = stage
            usd_path = os.path.join(os.path.dirname(__file__), "defs", "continuous_joints.usd")
            omni.usd.get_context().open_stage(usd_path)
    
    def build_ui(self):
        """
        Build a custom UI tool to run your extension.
        This function will be called any time the UI window is closed and reopened.
        """
        self.prepare_stage()

        self._robot_configuration_frame = CollapsableFrame("Robot Configuration", collapsed=False, enabled=True)

        with self._robot_configuration_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._selection_menu = DropDown(
                    "Select Articulation",
                    tooltip="Select from Articulations found on the stage after the timeline has been played.",
                    on_selection_fn=self._on_articulation_selection,
                    keep_old_selections=True,
                    # populate_fn = self._find_all_articulations # Equivalent functionality to one-liner below
                )
                # This sets the populate_fn to find all USD objects of a certain type on the stage, overriding the populate_fn arg
                # Figure out the type of an object with get_prim_object_type(prim_path)
                self._selection_menu.set_populate_fn_to_find_all_usd_objects_of_type("articulation", repopulate=False)

                self._stop_text = TextBlock(
                    "README",
                    "Select an Articulation and click the PLAY button on the left to get started.",
                    include_copy_button=False,
                    num_lines=2,
                )

                self._enable_ROS_2_button = StateButton(label="Real-Time Publish",
                                                   a_text="Enable",
                                                   b_text="Disable",
                                                   tooltip="Publish joint data in real-time to the real robot, this assumes that the robot is ready to subscribe to the messages to establish motor control",
                                                   on_a_click_fn=self._on_enable_publish_joint_data,
                                                   on_b_click_fn=self._on_disable_publish_joint_data)

                self._max_joint_velocity_adjustment_menu = CollapsableFrame("Max Joint Velocity", collapsed=True, enabled=False)

                def build_configuration_max_velocity_menu_fn():
                    self._max_joint_velocity_float_fields = []

                    # Don't build the frame unless there is a valid Articulation.
                    if self.articulation is None:
                        return

                    with ui.VStack(style=get_style(), spacing=5, height=0):
                        for i in range(self.articulation.num_dof):
                            field = FloatField(label=f"{self.articulation.dof_names[i]}", tooltip="Set max joint velocity")
                            field.set_on_value_changed_fn(
                                lambda value, index=i: self._on_set_max_joint_velocity(index, value)
                            )
                            self._max_joint_velocity_float_fields.append(field)
                    self._setup_max_joint_velocity_frame()
                
                self._max_joint_velocity_adjustment_menu.set_build_fn(build_configuration_max_velocity_menu_fn)


        
        self._robot_input_frame = CollapsableFrame("RL Control", collapsed=True, enabled=False)

        def build_robot_input_frame_fn():
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._input_field = StringField(label="Input", tooltip="Input to the LLM")
                submit_input_button = Button(label="Submit Input",
                                             text="Submit",
                                             tooltip="Input to the LLM",
                                             on_click_fn=self._on_submit_btn_click_fn)
                reach_target_button = Button(label="Reach Target",
                                             text="Reach",
                                             tooltip="Reach the target using the RL policy.",
                                             on_click_fn=self._on_reach_target_btn_click_fn)
                

        self._robot_input_frame.set_build_fn(build_robot_input_frame_fn)

        
        
        self._robot_control_frame = CollapsableFrame("Per Joint Control", collapsed=True, enabled=False)

        def build_robot_control_frame_fn():
            self._joint_control_frames = []
            self._joint_position_float_fields = []
            

            # Don't build the frame unless there is a valid Articulation.
            if self.articulation is None:
                return

            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._home_button = Button(label="Homing Button",
                                        text="Home",
                                        tooltip="Return the robot back to its original homing position.",
                                        on_click_fn=self._on_home_btn_click_fn)
                self._open_gripper_button = Button(label="Open Gripper Button",
                                        text="Open",
                                        tooltip="Opens the gripper of the robot arm.",
                                        on_click_fn=lambda: self._on_gripper_button_click_fn("open"))
                self._close_gripper_button = Button(label="Close Gripper Button",
                                        text="Close",
                                        tooltip="Closes the gripper of the robot arm.",
                                        on_click_fn=lambda: self._on_gripper_button_click_fn("close"))
                for i in range(self.articulation.num_dof):
                    field = FloatField(label=f"{self.articulation.dof_names[i]}", tooltip="Set joint position target")
                    field.set_on_value_changed_fn(
                        lambda value, index=i: self._on_set_joint_position_target(index, value)
                    )
                    self._joint_position_float_fields.append(field)

                self._publish_button = Button(label="Publish Joints",
                                              text="Publish",
                                              tooltip="Publish joint state to ROS 2",
                                              on_click_fn=self._on_publish_button_click_fn)
                
                save_action_button = Button(label="Save Positions",
                                            text="Save",
                                            tooltip="Save joint positions to XML file.",
                                            on_click_fn=self._on_save_action_button_click_fn)
                
                load_action_button = Button(label="Load Positions",
                                            text="Load",
                                            tooltip="Load joint positions from XML file.",
                                            on_click_fn=self._on_load_action_button_click_fn)

            self._setup_joint_control_frames()
        
        self._robot_control_frame.set_build_fn(build_robot_control_frame_fn)


        self._robot_logs_frame = CollapsableFrame("Logs", collapsed=True, enabled=False)

        def build_robot_logs_frame():
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._ros2_messages_field = TextBlock(label="ROS2", tooltip="Prints ROS 2 messages", num_lines=3, include_copy_button=True)
                self._llm_messages_field = TextBlock(label="LLM", tooltip="Prints LLM responses", num_lines=2, include_copy_button=True)
                self._isaac_sim_messages_field = TextBlock(label="Isaac Sim", tooltip="Prints Isaac Sim related messages", num_lines=2, include_copy_button=True)
                
                self._ros2_messages_field.set_text(self.ros2_messages)
                self._llm_messages_field.set_text(self.llm_messages)
                self._isaac_sim_messages_field.set_text(self.isaac_sim_messages)

        self._robot_logs_frame.set_build_fn(build_robot_logs_frame)


        self._robot_plots_frame = CollapsableFrame("Plots", collapsed=True, enabled=False)

        def build_robot_plots_frame():
            with self._robot_plots_frame:
                with ui.VStack(style=get_style(), spacing=5, height=0):
                    import numpy as np

                    x = np.arange(-1, 6.01, 0.01)
                    y = np.sin((x - 0.5) * np.pi)
                    plot = XYPlot(
                        "",
                        tooltip="Press mouse over the plot for data label",
                        x_data=[x[:300], x[100:400], x[200:]],
                        y_data=[y[:300], y[100:400], y[200:]],
                        x_min=None,  # Use default behavior to fit plotted data to entire frame
                        x_max=None,
                        y_min=-3.14,
                        y_max=3.14,
                        x_label="Time (s)",
                        y_label="Joint position (rad)",
                        plot_height=10,
                        legends=["x_joint", "y_joint", "z_joint"],
                        show_legend=True,
                        plot_colors=[
                            [255, 0, 0],
                            [0, 255, 0],
                            [0, 100, 200],
                        ],  # List of [r,g,b] values; not necessary to specify
                    )
            pass

        self._robot_plots_frame.set_build_fn(build_robot_plots_frame)


    ######################################################################################
    # Functions Below This Point Support The Provided Example And Can Be Replaced/Deleted
    ######################################################################################

    def _on_init(self):
        self.articulation = None

        # Initialize status messages
        self.ros2_messages = "ROS 2 Messages:\n"
        self.llm_messages = "LLM Responses:\n"
        self.isaac_sim_messages = "Isaac Sim Messages:\n"

        # Set omnigraph path
        self.GRAPH_PATH = "/World/ActionGraph"

        # Initialize ROS 2 bridge
        extensions.enable_extension('isaacsim.ros2.bridge')
        self.is_publish_joint_data = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = os.path.join(os.path.dirname(__file__), "policies", "position_only_policy.pt")
        # model_name = "2025-05-21_15-52-32"
        # model_path = f"/home/asimov/IsaacLab/logs/rsl_rl/franka_reach/{model_name}/exported/policy.pt"
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        # Initialize last actions for RL state collection
        self.last_actions = torch.tensor([0.0, 0.0, 0.0], device=self.device)

        self._reaching = False
        self._target_tolerance = 0.01

    def _robot_input_impulse(self, times=1):
        for i in range(times):
            og.Controller.set(
                    og.Controller.attribute("/World/ActionGraph/robot_input_impulse.state:enableImpulse"),
                    True
                )
    
    def _set_robot_input_value(self, value):
        og.Controller.edit(
                '/World/ActionGraph',
                {
                    og.Controller.Keys.SET_VALUES: [
                        ("/World/ActionGraph/robot_input_publisher.inputs:data", value)
                    ]
                }
            )

    def _invalidate_articulation(self):
        """
        This function handles the event that the existing articulation becomes invalid and there is
        not a new articulation to select.  It is called explicitly in the code when the timeline is
        stopped and when the DropDown menu finds no articulations on the stage.
        """
        self.articulation = None
        self._robot_input_frame.collapsed = True
        self._robot_input_frame.enabled = False
        self._robot_logs_frame.enabled = False
        self._robot_control_frame.collapsed = True
        self._robot_control_frame.enabled = False
        self._robot_plots_frame.collapsed = True
        self._robot_plots_frame.enabled = False
        self._max_joint_velocity_adjustment_menu.visible = False
        self._max_joint_velocity_adjustment_menu.enabled = False
        
    def _enable_articulation(self):
        self._robot_input_frame.collapsed = False
        self._robot_input_frame.enabled = True
        self._robot_logs_frame.enabled = True
        self._robot_control_frame.collapsed = False
        self._robot_control_frame.enabled = True
        self._robot_plots_frame.collapsed = False
        self._robot_plots_frame.enabled = True
        self._max_joint_velocity_adjustment_menu.visible = True
        self._max_joint_velocity_adjustment_menu.enabled = True

    def _on_articulation_selection(self, selection: str):
        """
        This function is called whenever a new selection is made in the
        "Select Articulation" DropDown.  A new selection may also be
        made implicitly any time self._selection_menu.repopulate() is called
        since the Articulation they had selected may no longer be present on the stage.

        Args:
            selection (str): The item that is currently selected in the drop-down menu.
        """
        # If the timeline is stopped, the Articulation won't be usable.
        if selection is None or self._timeline.is_stopped():
            self._invalidate_articulation()
            return

        self.articulation = SingleArticulation(selection)
        self.articulation.initialize()

        self._enable_articulation()
        self._robot_control_frame.rebuild()
        self._max_joint_velocity_adjustment_menu.rebuild()

    def _setup_max_joint_velocity_frame(self):
        num_dof = self.articulation.num_dof
        max_joint_velocities = self.articulation.dof_properties["maxVelocity"]

        for i in range(num_dof):
            field = self._max_joint_velocity_float_fields[i]

            max_joint_velocity = max_joint_velocities[i]

            field.set_value(max_joint_velocity)
            field.set_upper_limit(max_joint_velocity)
            field.set_lower_limit(0)

    def _setup_joint_control_frames(self):
        """
        Once a robot has been chosen, update the UI to match robot properties:
            Make a frame visible for each robot joint.
            Rename each frame to match the human-readable name of the joint it controls.
            Change the FloatField for each joint to match the current robot position.
            Apply the robot's joint limits to each FloatField.
        """
        num_dof = self.articulation.num_dof
        self.default_joint_positions = self.articulation.get_joint_positions()

        lower_joint_limits = self.articulation.dof_properties["lower"]
        upper_joint_limits = self.articulation.dof_properties["upper"]

        for i in range(num_dof):
            field = self._joint_position_float_fields[i]

            # Write the human-readable names of each joint
            position = self.default_joint_positions[i]

            field.set_value(position)
            field.set_upper_limit(upper_joint_limits[i])
            field.set_lower_limit(lower_joint_limits[i])

    def _on_set_joint_position_target(self, joint_index: int, position_target: float):
        """
        This function is called when the user changes one of the float fields
        to control a robot joint position target.  The index of the joint and the new
        desired value are passed in as arguments.

        This function assumes that there is a guarantee it is called safely.
        I.e. A valid Articulation has been selected and initialized
        and the timeline is playing. These guarantees are given by careful UI
        programming.  The joint control frames are only visible to the user when
        these guarantees are met.

        Args:
            joint_index (int): Index of robot joint that was modified
            position_target (float): New position target for robot joint
        """
        robot_action = ArticulationAction(
            joint_positions=np.array([position_target]),
            joint_velocities=np.array([0]),
            joint_indices=np.array([joint_index]),
        )
        self.articulation.apply_action(robot_action)

    def _on_set_max_joint_velocity(self, joint_index: int, max_joint_velocity: float):
        # TODO: set max joint velocity for the articulation joints
        pass

    def _on_enable_publish_joint_data(self):
        self.is_publish_joint_data = True
        self._publish_button.enabled = False
        og.Controller.edit(
            graph_path=self.GRAPH_PATH,
            changes={
                og.Controller.Keys.CONNECT: [
                    (f"/World/ActionGraph/on_playback_tick.outputs:tick", f"/World/ActionGraph/ros2_publish_joint_state.inputs:execIn")
                ]
            }
        )

    def _on_disable_publish_joint_data(self):
        self.is_publish_joint_data = False
        self._publish_button.enabled = True
        og.Controller.edit(
            graph_path=self.GRAPH_PATH,
            changes={
                og.Controller.Keys.DISCONNECT: [
                    (f"/World/ActionGraph/on_playback_tick.outputs:tick", f"/World/ActionGraph/ros2_publish_joint_state.inputs:execIn")
                ]
            }
        )

    def _on_home_btn_click_fn(self) -> None:
        for i in range(self.articulation.num_dof):
            field = self._joint_position_float_fields[i]

            # Reset robot control menu values to default
            position = self.default_joint_positions[i]
            field.set_value(position)

            robot_action = ArticulationAction(
                joint_positions=np.array([position]),
                joint_velocities=np.array([0]),
                joint_indices=np.array([i]),
            )
        
            self.articulation.apply_action(robot_action)

    def _on_gripper_button_click_fn(self, action):
        # TODO: open and close gripper
        if action == "open":
            pass
        if action == "close":
            pass

    def _on_submit_btn_click_fn(self):
        robot_input = self._input_field.get_value()

        if robot_input:
            self._set_robot_input_value(robot_input)
            self._robot_input_impulse()
        
        self._input_field.set_value("")

    def add_message_to_log(self, message_body: str, message: str):
        if message_body is self.ros2_messages:
            self.ros2_messages += message
            self._ros2_messages_field.set_text(self.ros2_messages)
        if message_body is self.llm_messages:
            self.llm_messages += message
            self._llm_messages_field.set_text(self.llm_messagesros2_messages)
        if message_body is self.isaac_sim_messages:
            self.isaac_sim_messages += message
            self._isaac_sim_messages_field.set_text(self.isaac_sim_messages)
                 
    def _on_publish_button_click_fn(self):
        og.Controller.set(
                    og.Controller.attribute("/World/ActionGraph/joint_state_impulse.state:enableImpulse"),
                    True
                )
    
    def get_prim_transformation(self, prim_path):

        stage = omni.usd.get_context().get_stage()

        if stage:

            prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
            
            # Check if prim exists and is transformable WARN: error starts here
            if prim.IsValid() and UsdGeom.Xformable(prim):
                xform = UsdGeom.Xformable(prim)
                transform_matrix = xform.GetLocalTransformation()

                # Extract translation
                translation = transform_matrix.ExtractTranslation() # pxr.Gf.Vec3d

                # Extract rotation as a quaternion
                rotation = transform_matrix.ExtractRotation().GetQuaternion() # pxr.Gf.Quaternion

            return translation, rotation

    def collect_state(self, position_only=True):
        current_joint_position = self.articulation.get_joint_positions()

        state_joint_position = torch.tensor(current_joint_position, device=self.device)

        # collect pose
        translation, rotation = self.get_prim_transformation("/World/Box")

        if position_only:
            pose = [translation[i] for i in range(len(translation))] + [1.0, 0.0, 0.0, 0.0]
        else:
            pose = [translation[i] for i in range(len(translation))] + [rotation[i] for i in range(len(rotation))]
            
        pose_command = torch.tensor(pose, device=self.device)

        # Concat state
        state = torch.cat((state_joint_position, pose_command, self.last_actions), dim=0)
        state = state.unsqueeze(0)

        return state

    def _on_reach_target_btn_click_fn(self):
        self._reaching = True

    def _on_save_action_button_click_fn(self):
        # Prepare directory
        save_dir = Path(__file__).parent / "saved_actions"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Random filename
        filename = "hello" + ".xml"
        filepath = save_dir / filename

        # Build raw XML string
        xml_data = "<joints>\n"
        for i in range(self.articulation.num_dof):
            joint_name = self.articulation.dof_names[i]
            joint_value = self._joint_position_float_fields[i].get_value()
            xml_data += f"  <{joint_name}>{joint_value}</{joint_name}>\n"
        xml_data += "</joints>"

        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(xml_data)

        print(f"Saved actions to {filepath}")

    def _on_load_action_button_click_fn(self):
        

    # def _find_all_articulations(self):
    # #    Commented code left in to help a curious user gain a thorough understanding

        #     import omni.usd
        #     from pxr import Usd
        #     items = []
        #     stage = omni.usd.get_context().get_stage()
        #     if stage:
        #         for prim in Usd.PrimRange(stage.GetPrimAtPath("/")):
        #             path = str(prim.GetPath())
        #             # Get prim type get_prim_object_type
        #             type = get_prim_object_type(path)
        #             if type == "articulation":
        #                 items.append(path)
        #     return items
