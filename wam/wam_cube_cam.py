import numpy as np
import time
import genesis as gs 
from genesis.utils.geom import trans_quat_to_T, xyz_to_quat, quat_to_T

########################## init ##########################
gs.init(backend=gs.gpu, precision="32")
########################## create a scene ##########################
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        res=(960, 640),
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    rigid_options=gs.options.RigidOptions(
        box_box_detection=True,
    ),
    show_viewer=True,
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)

wam = scene.add_entity(
    gs.morphs.URDF(
        file = 'urdf/wam_description/urdf/wam.urdf',
        fixed = True,
    )
)

cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.04, 0.04, 0.04),
        pos=(1.00, 0.0, 0.02),
    )
)
########################## cameras ##########################
cam_0 = scene.add_camera(
    # res=(1280, 960),
    fov=30,
    GUI=True,
)
########################## build ##########################
scene.build()

# fixed transformation
cam_0_transform = trans_quat_to_T(np.array([0.03, 0, 0.03]), xyz_to_quat(np.array([180+5, 0, -90])))

motors_dof = np.arange(15)
""" fingers_dof = np.arange(7, 9) """
""" end_effector = franka.get_link("hand") """
end_effector = wam.get_link("wam_link_7") #"wam_link_tcp"

#qpos = np.array([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04, 0, 0, 0, 0, 0, 0])
qpos = wam.inverse_kinematics(
    link=end_effector,
    pos=np.array([0, 1, 0]),
    quat=np.array([0, 1, 0, 0]),
)

# Only take the first 7 values for WAM's 7 DOFs
# qpos = qpos[:7]

""" print(f"qpos length: {len(qpos)}") """
""" print(f"{len(qs_idx)}") """
print("WAM DOFs:", wam.n_dofs)
print("WAM qpos size:", qpos.shape)

wam.set_qpos(qpos)
scene.step()
cam_0.set_pose(transform=trans_quat_to_T(end_effector.get_pos(), end_effector.get_quat()).cpu().numpy() @ cam_0_transform)
cam_0.render(rgb=True, depth=True)



wam.control_dofs_position(qpos, motors_dof)
""" franka.control_dofs_position(qpos[:-2], motors_dof)
franka.control_dofs_position([0.04, 0.04], fingers_dof)
 """
# hold
for i in range(100):
    print("hold", i)
    scene.step()
    cam_0.set_pose(transform=trans_quat_to_T(end_effector.get_pos(), end_effector.get_quat()).cpu().numpy() @ cam_0_transform)
    cam_0.render(rgb=True, depth=True)

# grasp
finder_pos = -0.0
for i in range(100):
    print("grasp", i)
    wam.control_dofs_position(qpos, motors_dof)
    # wam.control_dofs_position(np.array([finder_pos, finder_pos]), fingers_dof)
    scene.step()
    cam_0.set_pose(transform=trans_quat_to_T(end_effector.get_pos(), end_effector.get_quat()).cpu().numpy() @ cam_0_transform)
    cam_0.render(rgb=True, depth=True)

# lift
qpos = wam.inverse_kinematics(
    link=end_effector,
    pos=np.array([0.65, 0.0, 0.3]),
    quat=np.array([0, 1, 0, 0]),
)
for i in range(200):
    print("lift", i)
    wam.control_dofs_position(qpos, motors_dof)
    #wam.control_dofs_position(np.array([finder_pos, finder_pos]), fingers_dof)
    scene.step()
    cam_0.set_pose(transform=trans_quat_to_T(end_effector.get_pos(), end_effector.get_quat()).cpu().numpy() @ cam_0_transform)
    cam_0.render(rgb=True, depth=True)

# https://github.com/Genesis-Embodied-AI/Genesis/discussions/416