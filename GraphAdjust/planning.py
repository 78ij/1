from klampt.plan import robotcspace
from klampt.plan import cspace
from klampt.plan import robotplanning
from klampt.math import se3
from klampt import vis 
from klampt.io import resource
from klampt.model import ik
import klampt.model.create
from klampt.model import trajectory
import klampt.model.cartesian_trajectory
from klampt.model.collide import WorldCollider
from klampt import *
import time
import numpy as np
import glm
from copy import *
import sys

import os
from contextlib import contextmanager

@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different
from ignore import *
#settings: feel free to edit these to see how the results change
DO_SIMPLIFY = 0
DEBUG_SIMPLIFY = 0
MANUAL_SPACE_CREATION = 0
CLOSED_LOOP_TEST = 0
PLAN_TO_GOAL_TEST = 1
MANUAL_PLAN_CREATION = 0
def argmax(vec:glm.vec3):
  if(vec.x >= vec.y and vec.x >= vec.z): return 0
  if(vec.y >= vec.x and vec.y >= vec.z): return 1
  if(vec.z >= vec.x and vec.z >= vec.y): return 2
fn = "./reemc.urdf"
world_orig = WorldModel()
res = world_orig.readFile(fn)

def do_planning(line_data):
    #load the robot / world file
   # with stdout_redirected():


    LINE = 0
    ARC = 1
    class path_param:
        def __init__(self,type,moveable_id):
            self.type = type
            self.moveable_id = moveable_id
        def set_line(self,starting_point, ending_point):
            self.starting_point = starting_point
            self.ending_point = ending_point
        def set_arc(self,center,starting_point,radius,angle,axis):
            self.center = center
            self.starting_point = starting_point
            self.axis = axis
            self.radius = radius
            self.angle = angle
    params = []
    collision_objects = []
    moveable_id = None
    moveable_boxes = []
    idx = 0
    for l in line_data:
    #  print(line_data)
        list_l = l.split(' ')
        #print(list_l)
        if list_l[0] == 'path':
            if list_l[1] == 'line':
                param_t = path_param(LINE, moveable_id)
                param_t.set_line([float(list_l[2]),float(list_l[3]),float(list_l[4])],
                [float(list_l[5]),float(list_l[6]),float(list_l[7])])
                params.append(param_t)
            if list_l[1] == 'arc':
                param_t = path_param(ARC, moveable_id)
                param_t.set_arc([float(list_l[2]),float(list_l[3]),float(list_l[4])],
                [float(list_l[5]),float(list_l[6]),float(list_l[7])],
                float(list_l[8]),
                float(list_l[9]),
                [float(list_l[10]),float(list_l[11]),float(list_l[12])])
                params.append(param_t)
        if list_l[0] == 'box':
            pos = [float(list_l[1]),float(list_l[2]),float(list_l[3])]
            siz = [float(list_l[4]),float(list_l[5]),float(list_l[6])]
            ori = (float(list_l[7]),float(list_l[8]),float(list_l[9]),float(list_l[10]))
            # print(pos)
            # print(siz)
            # print(ori)
            # print(ori_so3)
            collision_objects.append([pos,siz,ori,int(list_l[11])])
        if list_l[0] == 'moveable':
            pos = [float(list_l[1]),float(list_l[2]),float(list_l[3])]
            siz = [float(list_l[4]),float(list_l[5]),float(list_l[6])]
            ori = (float(list_l[7]),float(list_l[8]),float(list_l[9]),float(list_l[10]))       
            # print(pos)
            # print(siz)
            # print(ori)
            # print(ori_so3)
            moveable_boxes.append([pos,siz,ori,int(list_l[11])])
            moveable_id = idx
            idx = idx + 1
    #for obj in collision_objects:
        
    #set up a settings dictionary here.  This is a random-restart + shortcutting
    #rrt planner.
    #ignore_list = []

    # ignore_list.append((link_1,link_2))
    ignore_list = []

    settings = { 'type':"sbl" }

    #This code generates a PRM with no specific endpoints
    #plan = cspace.MotionPlan(space, "prm", knn=10)
    #print "Planning..."
    #plan.planMore(500)
    #V,E = plan.getRoadmap()
    #print len(V),"feasible milestones sampled,",len(E),"edges connected"
    
    #klampt.model.create.primitives.box(0.3,0.3,0.3,center=[0.3,0.3,0.3],world=world)
    if not res:
        print("Unable to read file",fn)
        exit(0)
    percent_final = 0
    for param in params:
    # print('11111111111111111111111111111111111111111')
        world = world_orig.copy()
        
        print('----------doing planning-----------')

        robot = world.robot(0)
        resource.setDirectory("resources/"+robot.getName())
    # print("1111111111111111111Numlink: " + str(robot.numLinks()))
        for ign in IGNORE:
            l1 = ign[0]
            l2 = ign[1]
            link_1 = robot.link(l1)
            link_2 = robot.link(l2)
            if link_1 is None or link_2 is None:
                continue
        #  print(link_1.getIndex())
            #print(link_2.getIndex())
            if link_1.getIndex() == -1 or link_2.getIndex() == -1: continue
            robot.enableSelfCollision(link_1.getIndex(),link_2.getIndex(),False)
        vis.clear()
        co_2 = []
        param_moveable_idx = param.moveable_id
        moveable_box_tmp = moveable_boxes[param_moveable_idx]
        for b in collision_objects:
            if b[3] == moveable_box_tmp[3]: continue
            co_2.append(deepcopy(b))
        
        robot.setConfig([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        if param.type == LINE:
            starting = glm.vec3(param.starting_point[0],param.starting_point[1],param.starting_point[2])
            ending = glm.vec3(param.ending_point[0],param.ending_point[1],param.ending_point[2])
            offset = ending - starting
            angle = 0
            if(abs(offset.x) > abs(offset.y) and offset.x > 0):
                angle = 0.
            
            if(abs(offset.x) > abs(offset.y) and offset.x < 0):
                angle = 3.1415926
            
            if(abs(offset.x) < abs(offset.y) and offset.y > 0):
                angle = 3.1415926 / 2
            
            if(abs(offset.x) < abs(offset.y) and offset.y < 0):
                angle = -3.1415926 / 2

            offset = glm.normalize(offset)
            offset = offset * 0.3
            offset2 = offset * 0.1
            offsett = starting + offset
            starting.x -= offsett.x
            starting.y -= offsett.y

            ending.x -= offsett.x
            ending.y -= offsett.y
        # print('offsett' + str(offsett))
            for i in range(len(co_2)):
                co_2[i][0][0] -= offsett.x
                co_2[i][0][1] -= offsett.y

            moveable_box_tmp[0][0] -= offsett.x
            moveable_box_tmp[0][1] -= offsett.y
            quat = glm.angleAxis(angle,glm.vec3(0,0,1))
            starting = quat * starting
            ending = quat * ending
            for co in co_2:
                position = glm.vec3(co[0][0],co[0][1],co[0][2])
                position = quat * position
                quat_orig = glm.quat(co[2][0],co[2][1],co[2][2],co[2][3])
                quat2 = quat * quat_orig
                so3_rot = klampt.math.so3.from_quaternion((quat2.w,quat2.x,quat2.y,quat2.z))
                b = klampt.model.create.primitives.box(co[1][0],co[1][1],co[1][2],
                center=[0.,0.,0.],
                R = so3_rot,t=[position.x,position.y,position.z],world=world)
            pos_mov = glm.vec3(moveable_box_tmp[0][0],moveable_box_tmp[0][1],moveable_box_tmp[0][2])
            pos_mov = quat * pos_mov
            quat_mov_orig = glm.quat(moveable_box_tmp[2][0],moveable_box_tmp[2][1],moveable_box_tmp[2][2],moveable_box_tmp[2][3])
            quat_mov2 = quat * quat_mov_orig
            mov_so3_rot = klampt.math.so3.from_quaternion((quat_mov2.w,quat_mov2.x,quat_mov2.y,quat_mov2.z))
            moveable_geom = klampt.model.create.primitives.box(moveable_box_tmp[1][0],moveable_box_tmp[1][1],moveable_box_tmp[1][2],
                center=[0.,0.,0.],R= mov_so3_rot,t=[pos_mov.x,pos_mov.y,pos_mov.z],world=world)
            hand_direction = ending - starting
            starting.x = starting.x - 0.2
            if(starting.z > 1):
                pose = glm.quat(0.7071068,0.7071068,0,0)
            else:
                pose = glm.quat(0.7071068,-0.7071068,0,0)
            pose_R = klampt.math.so3.from_quaternion((pose.w,pose.x,pose.y,pose.z))
            wholepath = [robot.getConfig()]
            
            initial_objective = klampt.model.ik.objective(robot.link('hand_right_palm_link'),
            R = pose_R,
            t = [starting.x,starting.y,starting.z],world=world)
        
            # vis.clear()
            # for i in range(0,world.numRobots()):
            #     vis.add("robot",world.robot(i))
            # for i in range(world.numRigidObjects()):
            #     vis.add("rigidObject"+str(i),world.rigidObject(i))
            # for i in range(world.numTerrains()):
            #     vis.add("terrain"+str(i),world.terrain(i))
            # vis.dialog()
            #start from end of previous path
            try:

                plan = robotplanning.planToCartesianObjective(world,robot,[initial_objective],movingSubset='all',ignoreCollisions=ignore_list,**settings)
                t0 = time.time()
                plan.space.cspace.enableAdaptiveQueries(True)
            # print("Planning...")
                for round in range(5):
                    print('s')
                    plan.planMore(50)
                    path = plan.getPath()
                    if path is not None and len(path) > 0: break
                print("Planning time, 500 iterations",time.time()-t0)
                #this code just gives some debugging information. it may get expensive
                path = plan.getPath()
                wholepath += path[1:]
            except:
                percent = 0
                percent_final += percent
               # plan.space.close()
               # plan.close()
                continue

            #to be nice to the C++ module, do this to free up memory
            plan.space.close()
            plan.close()
            rigidmodelnum = world.numTerrains()
        #  print('num: ' + str(rigidmodelnum))
            if len(wholepath)>1:
        #print "Path:"
        #for q in wholepath:
        #    print "  ",q
        #if you want to save the path to disk, uncomment the following line
        #wholepath.save("test.path")

        #draw the path as a RobotTrajectory (you could just animate wholepath, but for robots with non-standard joints
        #the results will often look odd).  Animate with 5-second duration
                times = [i*5.0/(len(wholepath)-1) for i in range(len(wholepath))]
                traj = trajectory.RobotTrajectory(robot,times=times,milestones=wholepath)
                #show the path in the visualizer, repeating for 60 seconds
            
                # vis.animate("robot",traj)
                # vis.spin(60)
                #vis.pauseAnimation()
                configx = wholepath[-1]
                #print(configx)
                configx[-4] = 0
                robot.setConfig(configx)
                link = robot.link('hand_right_grasping_frame')
                world.remove(moveable_geom)
                
                moveable_geom = klampt.model.create.primitives.box(moveable_box_tmp[1][0],moveable_box_tmp[1][1],moveable_box_tmp[1][2],
                center=[0.,0.,0.],
                )
                lgeom = link.geometry()
            # print(lgeom)
                lxform = link.getTransform()
            # ogeom= moveable_geom.()#dont modify object’s actual geometry
                oxform = ( mov_so3_rot,[pos_mov.x,pos_mov.y,pos_mov.z])
                relxform = se3.mul(se3.inv(lxform),oxform)
                moveable_geom.transform(*relxform)
                group = Geometry3D()
                group.setGroup()
                group.setElement(0,moveable_geom)
                group.setElement(1,lgeom)
                link.geometry().set(group)
            

            initconfig = wholepath[-1]
            wholepath = []
            milestones_interp = []
            for step in range(30):
            # print(step)
                point = (ending-starting) / 30. *float(step) + starting
                #initial_objective = klampt.model.ik.objective(robot.link('hand_right_grasping_frame'),
                #R = pose_R,
                #t = [pos.x,pos.y,pos.z])
                milestones_interp.append((deepcopy(pose_R),[deepcopy(point.x),deepcopy(point.y),deepcopy(point.z)]))
            cspace_t = klampt.plan.robotplanning.makeSpace(world,robot)

            traj_interp = klampt.model.trajectory.SE3Trajectory(milestones=milestones_interp)
            traj_f = klampt.model.cartesian_trajectory.cartesian_path_interpolate(robot,traj_interp,'hand_right_grasping_frame',startConfig=initconfig,feasibilityTest=cspace_t.feasible)
            if isinstance(traj_f,tuple):
                percent = traj_f[0]
                traj_f = traj_f[1]
            else:
                percent = 1
            cspace_t.close()
            print('percent: ' + str(percent))
            percent_final += percent
        #  vis.animate("robot",traj2)
        #  vis.spin(60)   
        #  print(wholepath)
            link.geometry().set(lgeom)





        if param.type == ARC:
        #  print('2222222222222222222222222222222222222222')
            starting = glm.vec3(param.starting_point[0],param.starting_point[1],param.starting_point[2])
            center = glm.vec3(param.center[0],param.center[1],param.center[2])
            axis = glm.vec3(param.axis[0],param.axis[1],param.axis[2])
        #   print('starting ' + str(starting))
            ending = glm.rotate(starting - center, param.angle,axis) + center
            offset = glm.cross(starting - center, axis)
            start_to_end = ending - starting
            if glm.dot(start_to_end,offset) < 0 : offset = -offset
            
            angle = 0
            if(abs(offset.x) > abs(offset.y) and offset.x > 0):
                angle = 0.
            
            if(abs(offset.x) > abs(offset.y) and offset.x < 0):
                angle = 3.1415926
            
            if(abs(offset.x) < abs(offset.y) and offset.y > 0):
                angle = 3.1415926 / 2
            
            if(abs(offset.x) < abs(offset.y) and offset.y < 0):
                angle = -3.1415926 / 2
            
            offset = glm.normalize(offset)
            offset = offset * min(0.3,param.radius)
            offset2 = offset * 0.1
            offsett = starting + offset
            starting.x -= offsett.x
            starting.y -= offsett.y

            center.x -= offsett.x
            center.y -= offsett.y

            ending.x -= offsett.x
            ending.y -= offsett.y
        #    print('offsett' + str(offsett))
            for i in range(len(co_2)):
                co_2[i][0][0] -= offsett.x
                co_2[i][0][1] -= offsett.y

            moveable_box_tmp[0][0] -= offsett.x
            moveable_box_tmp[0][1] -= offsett.y
            quat = glm.angleAxis(angle,glm.vec3(0,0,1))
            starting = quat * starting
            ending = quat * ending
            axis = quat * axis
            center = quat * center

            for co in co_2:
                position = glm.vec3(co[0][0],co[0][1],co[0][2])
                position = quat * position
                quat_orig = glm.quat(co[2][0],co[2][1],co[2][2],co[2][3])
                quat2 = quat * quat_orig
                so3_rot = klampt.math.so3.from_quaternion((quat2.w,quat2.x,quat2.y,quat2.z))
                b = klampt.model.create.primitives.box(co[1][0],co[1][1],co[1][2],
                center=[0.,0.,0.],
                R = so3_rot,t=[position.x,position.y,position.z],world=world)
            pos_mov = glm.vec3(moveable_box_tmp[0][0],moveable_box_tmp[0][1],moveable_box_tmp[0][2])
            pos_mov = quat * pos_mov
            quat_mov_orig = glm.quat(moveable_box_tmp[2][0],moveable_box_tmp[2][1],moveable_box_tmp[2][2],moveable_box_tmp[2][3])
            quat_mov2 = quat * quat_mov_orig
            mov_so3_rot = klampt.math.so3.from_quaternion((quat_mov2.w,quat_mov2.x,quat_mov2.y,quat_mov2.z))
            moveable_geom = klampt.model.create.primitives.box(moveable_box_tmp[1][0],moveable_box_tmp[1][1],moveable_box_tmp[1][2],
                center=[0.,0.,0.],R= mov_so3_rot,t=[pos_mov.x,pos_mov.y,pos_mov.z],world=world)
        
        # relxform = se3.mul(se3.inv(lxform),oxform)
        # ogeom.transform(*relxform)
        # group = Geometry3D()
            #group.setGroup()
        # group.setElement(0,ogeom)
        #   group.setElement(1,ogeom)

            # for i in range(0,world.numRobots()):
            #     vis.add("robot",world.robot(i))
            # for i in range(world.numRigidObjects()):
            #     vis.add("rigidObject"+str(i),world.rigidObject(i))
            # for i in range(world.numTerrains()):
            #     vis.add("terrain"+str(i),world.terrain(i))
            # print(len(robot.getConfig()))
        #  vis.dialog()

            #Generate a path connecting the edited configurations
            #You might edit the value 500 to play with how many iterations to give the
            #planner.
            # for i in range(1,world.numRobots()):
            #     vis.add("robot"+str(i),world.robot(i))
            # for i in range(world.numRigidObjects()):
            #     vis.add("rigidObject"+str(i),world.rigidObject(i))
            # for i in range(world.numTerrains()):
            #     vis.add("terrain"+str(i),world.terrain(i))
            # vis.dialog()
            hand_direction = starting - center
            starting.x = starting.x - 0.2
            maxidx = argmax(glm.abs(hand_direction))
            if maxidx == 0:
                if hand_direction.x >= 0:
                    pose = glm.quat(0,0,0.7071068,0.7071068)
                else:
                    pose = glm.quat(0.7071068,0,0,0.7071068)
            if maxidx == 1:
                if hand_direction.y >= 0:
                    pose = glm.quat(1,0,0,0)
                else:
                    pose = glm.quat(0,1,0,0)
            if maxidx == 2:
                    pose = glm.quat(0.7071068,0.7071068,0,0)
            pose_R = klampt.math.so3.from_quaternion((pose.w,pose.x,pose.y,pose.z))
            wholepath = [robot.getConfig()]
            
            initial_objective = klampt.model.ik.objective(robot.link('hand_right_palm_link'),
            R = pose_R,
            t = [starting.x,starting.y,starting.z],world=world)
        
            try:
                #start from end of previous path
                plan = robotplanning.planToCartesianObjective(world,robot,[initial_objective],movingSubset='all',ignoreCollisions=ignore_list,**settings)
                t0 = time.time()
                plan.space.cspace.enableAdaptiveQueries(True)
            #   print("Planning...")
                for round in range(5):
                # print('s')
                    plan.planMore(50)
                    path = plan.getPath()
                    if path is not None and len(path) > 0: break
            #  print("Planning time, 500 iterations",time.time()-t0)
                #this code just gives some debugging information. it may get expensive
                path = plan.getPath()
                wholepath += path[1:]
            except:
                percent = 0
                percent_final += percent
               # plan.space.close()
              #  plan.close()
                continue


            #to be nice to the C++ module, do this to free up memory
            plan.space.close()
            plan.close()
            rigidmodelnum = world.numTerrains()
        # print('num: ' + str(rigidmodelnum))
            if len(wholepath)>1:
        #print "Path:"
        #for q in wholepath:
        #    print "  ",q
        #if you want to save the path to disk, uncomment the following line
        #wholepath.save("test.path")

        #draw the path as a RobotTrajectory (you could just animate wholepath, but for robots with non-standard joints
        #the results will often look odd).  Animate with 5-second duration
                times = [i*5.0/(len(wholepath)-1) for i in range(len(wholepath))]
                traj = trajectory.RobotTrajectory(robot,times=times,milestones=wholepath)
                #show the path in the visualizer, repeating for 60 seconds
            
            #   vis.animate("robot",traj)
            #   vis.spin(60)
                #vis.pauseAnimation()
                configx = wholepath[-1]
                #print(configx)
                configx[-4] = 0
                robot.setConfig(configx)
                link = robot.link('hand_right_grasping_frame')
                world.remove(moveable_geom)
                
                moveable_geom = klampt.model.create.primitives.box(moveable_box_tmp[1][0],moveable_box_tmp[1][1],moveable_box_tmp[1][2],
                center=[0.,0.,0.],
                )
                lgeom = link.geometry()
            # print(lgeom)
                lxform = link.getTransform()
            # ogeom= moveable_geom.()#dont modify object’s actual geometry
                oxform = ( mov_so3_rot,[pos_mov.x,pos_mov.y,pos_mov.z])
                relxform = se3.mul(se3.inv(lxform),oxform)
                moveable_geom.transform(*relxform)
                group = Geometry3D()
                group.setGroup()
                group.setElement(0,moveable_geom)
                group.setElement(1,lgeom)
                link.geometry().set(group)
            
        #   vis.clear()
            # for i in range(0,world.numRobots()):
            #     vis.add("robot",world.robot(i))
            # for i in range(world.numRigidObjects()):
            #     vis.add("rigidObject"+str(i),world.rigidObject(i))
            # for i in range(world.numTerrains()):
            #     vis.add("terrain"+str(i),world.terrain(i))
        #   vis.dialog()
            initconfig = wholepath[-1]
            wholepath = []
            milestones_interp = []
            for step in range(30):
            # print(step)
                quat = pose
                quat = glm.angleAxis(param.angle * step / 30, axis) * quat
                pos = glm.rotate(starting - center, param.angle * step / 30,axis) + center
                pose_R = klampt.math.so3.from_quaternion((quat.w,quat.x,quat.y,quat.z))
                #initial_objective = klampt.model.ik.objective(robot.link('hand_right_grasping_frame'),
                #R = pose_R,
                #t = [pos.x,pos.y,pos.z])
                milestones_interp.append((deepcopy(pose_R),[deepcopy(pos.x),deepcopy(pos.y),deepcopy(pos.z)]))
            cspace_t = klampt.plan.robotplanning.makeSpace(world,robot)

            traj_interp = klampt.model.trajectory.SE3Trajectory(milestones=milestones_interp)
            traj_f = klampt.model.cartesian_trajectory.cartesian_path_interpolate(robot,traj_interp,'hand_right_grasping_frame',startConfig=initconfig,feasibilityTest=cspace_t.feasible)
            if isinstance(traj_f,tuple):
                percent = traj_f[0]
                traj_f = traj_f[1]
            else:
                percent = 1
            print('percent:' + str(percent))
            percent_final += percent
            cspace_t.close()
        #  vis.animate("robot",traj2)
        #  vis.spin(60)   
        #  print(wholepath)
            link.geometry().set(lgeom)
    print('--------------------final' + str(percent_final))
    return float(percent_final) / len(params)
    #vis.kill()
print('finished!')
