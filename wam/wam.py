import genesis as gs
import time
import numpy as np
from genesis.utils.geom import trans_quat_to_T, xyz_to_quat, quat_to_T

gs.init(backend=gs.gpu)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        # camera_pos=(3, -3, 1.5),
        camera_pos=(2.5, -2, 1),
        camera_lookat=(1.0, 0.0, 0.5),
        camera_fov=30,
        res=(960, 640),
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    rigid_options=gs.options.RigidOptions(
        box_box_detection=True,
        enable_collision=True,
    ),
    show_viewer=True,
)

plane = scene.add_entity(
    gs.morphs.Plane(),
)

wam = scene.add_entity(
    gs.morphs.URDF(
        file = './wam_description/urdf/wam_finger.urdf',
        pos = (0, 0, 0),
        fixed = True,
    )
)

cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.08, 0.08, 0.08),
        pos=(1.00, 0.0, 0.05),
        collision=True,
    )
)

# cylinder = scene.add_entity(
#     gs.morphs.Cylinder(
#         height = 0.07,
#         radius = 0.05,
#         pos  = (1.00, 0.0, 0.02),
#         quat = (0, 0, 1, 0)
#     )
# )

cam = scene.add_camera(
    res=(640, 480),
    pos=(2.5, -2, 1),
    lookat=(1, 0, 0.5),
    fov=30,
    GUI=True,
)

scene.build()

# rgb = cam.render(rgb=True)

# cam.start_recording()

# for link in wam.links:
#     print(link.name)

arm_jnt_names = [
    "wam_joint_1",
    "wam_joint_2",
    "wam_joint_3",
    "wam_joint_4",
    "wam_joint_5",
    "wam_joint_6",
]

arms_dofs_idx = [wam.get_joint(name).dof_idx_local for name in arm_jnt_names]

wam.set_dofs_kp(
    kp = np.array([300 for _ in range(6)]),
    dofs_idx_local = arms_dofs_idx,
)
wam.set_dofs_kv(
    kv = np.array([400 for _ in range(6)]),
    dofs_idx_local = arms_dofs_idx,
)
wam.set_dofs_force_range(
    lower = np.array([-30 for _ in range(6)]),
    upper = np.array([ 30 for _ in range(6)]),
    dofs_idx_local = arms_dofs_idx,
)

end_effector = wam.get_link("wam_link_7") #"wam_link_tcp"
# Offset to compensate for missing wam_link_tcp
tcp_offset = np.array([0.0, 0.0, 0.06])

# Close fingers to grasp
hand_jnt_names = [
    "bhand_finger1",
    "bhand_finger2",
    "bhand_finger3"
]
hand_dofs_idx = [wam.get_joint(name).dof_idx_local for name in hand_jnt_names]

# Set gains for the hand (adjust values as needed)
wam.set_dofs_kp(
    kp=np.array([4000 for _ in range(3)]),  # Lower KP since fingers need less force
    dofs_idx_local=hand_dofs_idx,
)
wam.set_dofs_kv(
    kv=np.array([400 for _ in range(3)]),  # Lower KV for smoother grasping
    dofs_idx_local=hand_dofs_idx,
)
wam.set_dofs_force_range(
    lower=np.array([-10 for _ in range(3)]),  # Limit force to prevent over-gripping
    upper=np.array([5 for _ in range(3)]),
    dofs_idx_local=hand_dofs_idx,
)

grasp_pos = np.array([2.2, 2.2, 2.2])

scene.step()

target_pos = np.array([1, 0, 0.25]) + tcp_offset  # desired (x, y ,z)
target_quat = np.array([0, 1, 0, 0]) #(w, x, y ,z)
qpos = wam.inverse_kinematics(
    link=end_effector, 
    pos=target_pos, 
    quat=target_quat
)

# Step 3: Close fingers to grasp
for i in range(400):
    print(f"Moving to cube {i}")
    wam.control_dofs_position(qpos[:7], np.arange(7))
    scene.step()
    # cam.render()

for i in range(200):
    print(f"Closing fingers {i}")
    wam.control_dofs_position(grasp_pos, hand_dofs_idx)
    scene.step()
    # cam.render()

# Step 4: Lift the cube
target_pos = np.array([1, 0, 1]) + tcp_offset
target_quat = np.array([0, 1, 0, 0]) #(w, x, y ,z)
qpos = wam.inverse_kinematics(link=end_effector, pos=target_pos, quat=target_quat)

for i in range(600):
    print(f"Lifting {i}")
    wam.control_dofs_position(qpos[:7], np.arange(7))  # Move arm
    wam.control_dofs_position(grasp_pos, hand_dofs_idx)  # Maintain grasp force
    scene.step()

    # cam.render()

# cam.stop_recording(save_to_filename="wam_nospread.mp4", fps=60)