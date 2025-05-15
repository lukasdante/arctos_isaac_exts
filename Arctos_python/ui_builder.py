# Copyright (c) 2022-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import numpy as np
import omni.timeline
import omni.ui as ui
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.prims import get_prim_object_type
from isaacsim.core.utils.types import ArticulationAction, ArticulationActions
from isaacsim.gui.components.element_wrappers import (
    CollapsableFrame, DropDown, FloatField, TextBlock, StateButton, Button, StringField)
from isaacsim.gui.components.ui_utils import get_style


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
        elif self._timeline.is_stopped():
            self._stop_text.visible = True

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
        elif event.type == int(omni.usd.StageEventType.SIMULATION_STOP_PLAY):  # Timeline stopped
            # Ignore pause events
            if self._timeline.is_stopped():
                self._invalidate_articulation()
                self._stop_text.visible = True

    def cleanup(self):
        """
        Called when the stage is closed or the extension is hot reloaded.
        Perform any necessary cleanup such as removing active callback functions
        Buttons imported from isaacsim.gui.components.element_wrappers implement a cleanup function that should be called
        """
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()

    def build_ui(self):
        """
        Build a custom UI tool to run your extension.
        This function will be called any time the UI window is closed and reopened.
        """
        # TODO: logs_frame = ScrollingFrame("ROS 2 and Isaac Sim Logs")

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

                self._enable_ROS_2_button = StateButton(label="ROS 2 Enable Button",
                                                   a_text="Enable",
                                                   b_text="Disable",
                                                   tooltip="Enable ROS 2 Bridge for Hardware-in-Loop (HIL)",
                                                   on_a_click_fn=self._on_enable_ROS2_button_click_fn,
                                                   on_b_click_fn=self._on_disable_ROS2_button_click_fn)

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
        
        def build_robot_configuration_fn():
            # add button for Enable ROS 2
            # add home button
            pass

        self._robot_configuration_frame.set_build_fn(build_robot_configuration_fn)
        
        self._robot_input_frame = CollapsableFrame("Robot Input", collapsed=True, enabled=False)

        def build_robot_input_frame_fn():
            with ui.VStack(style=get_style(), spacing=5, height=0):
                input_field = StringField(label="Input", tooltip="Input to the LLM")
                

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
            self._setup_joint_control_frames()
        

        self._robot_control_frame.set_build_fn(build_robot_control_frame_fn)

        self._robot_plots_frame = CollapsableFrame("Robot Plots", collapsed=True, enabled=False)

        def build_robot_plots_frame():
            pass

        self._robot_plots_frame.set_build_fn(build_robot_plots_frame)

    ######################################################################################
    # Functions Below This Point Support The Provided Example And Can Be Replaced/Deleted
    ######################################################################################

    def _on_init(self):
        self.articulation = None

    def _invalidate_articulation(self):
        """
        This function handles the event that the existing articulation becomes invalid and there is
        not a new articulation to select.  It is called explicitly in the code when the timeline is
        stopped and when the DropDown menu finds no articulations on the stage.
        """
        self.articulation = None
        # TODO: add other fields for robot config to become invisible
        self._robot_input_frame.collapsed = True
        self._robot_input_frame.enabled = False
        self._robot_control_frame.collapsed = True
        self._robot_control_frame.enabled = False
        self._robot_plots_frame.collapsed = True
        self._robot_plots_frame.enabled = False
        self._max_joint_velocity_adjustment_menu.visible = False
        self._max_joint_velocity_adjustment_menu.enabled = False
        
    def _enable_articulation(self):
        # TODO: add other fields for robot config to become visible
        self._robot_input_frame.collapsed = False
        self._robot_input_frame.enabled = True
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

    def _setup_robot_input_frame(self):
        pass

    def _setup_robot_configuration_frame(self):
        pass

    def _setup_robot_plots_frame(self):
        pass

    def _setup_joint_control_frames(self):
        """
        Once a robot has been chosen, update the UI to match robot properties:
            Make a frame visible for each robot joint.
            Rename each frame to match the human-readable name of the joint it controls.
            Change the FloatField for each joint to match the current robot position.
            Apply the robot's joint limits to each FloatField.
        """
        num_dof = self.articulation.num_dof
        joint_positions = self.articulation.get_joint_positions()

        lower_joint_limits = self.articulation.dof_properties["lower"]
        upper_joint_limits = self.articulation.dof_properties["upper"]

        for i in range(num_dof):
            field = self._joint_position_float_fields[i]

            # Write the human-readable names of each joint
            # frame.title = dof_names[i]
            position = joint_positions[i]

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

    def _on_enable_ROS2_button_click_fn(self):
        # TODO: enable ROS 2
        pass

    def _on_disable_ROS2_button_click_fn(self):
        # TODO: disable ROS 2
        pass

    def _on_home_btn_click_fn(self) -> None:
        # TODO: obtain default position
        for i in range(self.articulation.num_dof):
            robot_action = ArticulationAction(
                joint_positions=np.array([0]),
                joint_velocities=np.array([0]),
                joint_indices=np.array([i]),
            )
        
            self.articulation.apply_action(robot_action)

            # TODO: change value of the affected joints to 0

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
