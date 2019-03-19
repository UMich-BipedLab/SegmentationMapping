#!/usr/bin/python

import rospy, sys, os
import rosbag
import geometry_msgs, tf
import tf.msg
import geometry_msgs.msg
import pdb

def gen_tf_msg_from_file(fname):
    msgs = []

    with open(fname, 'r') as f: 
        for i, line in enumerate(f):
            words = line.split(',')
            if i == 0:
                origin = []
                origin.append( float(words[1]))
                origin.append ( float(words[2]))
                origin.append( float(words[3]))
            

            time = int(words[0])
            secs = time // 1000000
            nsecs = (time - secs * 1000000) * 1000
            x = float(words[1])
            y = float(words[2])
            z = float(words[3])
            R = float(words[4])
            P = float(words[5])
            Y = float(words[6])
            # convert to local coordinate frame, located at the origin 
            x = x - origin[0]
            y = y - origin[1]
            z = z - origin[2]
            # create TF msg 
            tf_msg = tf.msg.tfMessage()
            geo_msg = geometry_msgs.msg.TransformStamped()
            geo_msg.header.stamp = rospy.Time(secs,nsecs)
            geo_msg.header.seq = i
            geo_msg.header.frame_id = "map"
            geo_msg.child_frame_id = "body"
            geo_msg.transform.translation.x = x
            geo_msg.transform.translation.y = y
            geo_msg.transform.translation.z = z
            angles = tf.transformations.quaternion_from_euler(R,P,Y) 
            geo_msg.transform.rotation.x = angles[0]
            geo_msg.transform.rotation.y = angles[1]
            geo_msg.transform.rotation.z = angles[2]
            geo_msg.transform.rotation.w = angles[3]
            tf_msg.transforms.append(geo_msg)
            msgs.append(tf_msg)
    return msgs

def write_bagfile(data_set_name, tf_msgs ):
    ''' Write all ROS msgs, into the bagfile. 
    '''
    bag = rosbag.Bag(data_set_name, 'w')
    # write the tf msgs into the bagfile
    for msg in tf_msgs:
        print(msg.transforms[0].header.stamp)
        bag.write('/tf', msg, msg.transforms[0].header.stamp )
    bag.close()
    print "Finished Bagfile IO"


if __name__ == "__main__":
    gt_pose_fname = sys.argv[1]
    output_bag = sys.argv[2]
    print("Gen tf msgs")
    msgs = gen_tf_msg_from_file(gt_pose_fname)
    print("Write bag file")
    write_bagfile(output_bag, msgs)
