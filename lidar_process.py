#!/usr/bin/env python

import rospy
import math
import numpy as np
from sensor_msgs.msg import LaserScan

class Lidar:
    def __init__(self):

        self.lidar_data = LaserScan()
        self.filtered_data = np.zeros(360)
        self.lidar_max = 3.5        # Maximum range of LiDAR
        self.candidates = []
        self.lidar_scan_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.newCandts = []

        self.yaw2goal_thresh = 5.0


    def lidar_callback(self, data):
        self.lidar_data = data

        for i in range(len(self.lidar_data.ranges)):
            if self.lidar_data.ranges[i] == 0.0 or self.lidar_data.ranges[i] == np.inf:
                self.filtered_data[i] = self.lidar_max
            else:
                self.filtered_data[i] = self.lidar_data.ranges[i]


    def exact_possible(self, split=5, dist=1.0, limit=0.73, minDist=0.1, ret=False):
        moving_dist = np.linspace(limit, dist+limit, split)
        temp = self.scan_possible(dist=moving_dist[0], limit=limit, ret=True)
        for i in range(len(moving_dist)):
            temp2 = self.scan_possible_sub(possibles=temp, dist=moving_dist[i], limit=limit)
            if len(temp2) == 0:
                break
            else:
                temp = temp2

        if len(temp) == 0:
            temp = self.scan_possible(dist=minDist, limit=limit, ret=True)

        degs = temp[:][0]
        degsNew = np.where(degs<=180, degs, degs-360)
        canditsNew = np.array([np.deg2rad(degsNew), temp[:][1]])
        self.newCandts = canditsNew

        if ret:
            return canditsNew



    def scan_possible_sub(self, possibles, dist, limit):
        temp = np.arctan2(dist, limit)
        theta = np.pi / 2 - temp
        theta_deg = np.rad2deg(theta)
        theta_int = int(math.ceil(theta_deg))

        possible_head = []
        for i in range(len(possibles)):
            idx = possibles[i, 0]
            if idx < theta_int:
                scan_pos = self.filtered_data[:theta_int + idx + 1]
                scan_neg = self.filtered_data[-(theta_int - idx):]
                scan = np.concatenate((scan_neg, scan_pos), axis=None)

            elif (idx >= theta_int) and (idx < len(self.filtered_data) - theta_int):
                scan = self.filtered_data[(idx - theta_int):(idx + theta_int + 1)]

            else:
                scan_pos = self.filtered_data[(idx - theta_int):]
                scan_neg = self.filtered_data[:theta_int - (len(self.filtered_data) - idx) + 1]
                scan = np.concatenate((scan_pos, scan_neg), axis=None)

            min_dist = np.amin(scan)
            if min_dist >= dist + limit:
                possible_head.append([i, min_dist])


        return np.array(possible_head)






    def scan_possible(self, dist=1.0, limit=0.73, ret=False, rad=False):
        # dist: moving distance for next step
        # limit: width of drone + margin
        # ret: if true, it returns candidates as numpy array
        # output: n x 2 numpy array, each row: [direction without collision (radian) from drone coordinate, moving distance without collision (m)]
        temp = np.arctan2(dist, limit)
        theta = np.pi/2 - temp
        theta_deg = np.rad2deg(theta)
        theta_int = int(math.ceil(theta_deg))

        possible_head = []
        possible_head_idx = []
        for i in range(len(self.lidar_data.ranges)):
            if i < theta_int:
                scan_pos = self.filtered_data[:theta_int+i+1]
                scan_neg = self.filtered_data[-(theta_int-i):]
                scan = np.concatenate((scan_neg, scan_pos), axis=None)

            elif (i >= theta_int) and (i < len(self.lidar_data.ranges)-theta_int):
                scan = self.filtered_data[(i-theta_int):(i+theta_int+1)]

            else:
                scan_pos = self.filtered_data[(i-theta_int):]
                scan_neg = self.filtered_data[:theta_int-(len(self.lidar_data.ranges)-i)+1]
                scan = np.concatenate((scan_pos, scan_neg), axis=None)

            min_dist = np.amin(scan)
            newI = 0
            if min_dist >= dist+limit:
                if i > 180:
                    newI = i - 360
                possible_head.append([np.rad2deg(newI), min_dist])
                possible_head_idx.append([i, min_dist])

        self.candidates = np.array(possible_head)

        if ret:
            return np.array(possible_head)
        if ret and rad == False:
            return np.array(possible_head_idx)

    # self.desired_pos.pose.position.x = self.local_position.pose.position.x
    def next_Best(self, waypoint, curPos, dist=1.0, relative=False):
        # waypoint: Target x, y position from world coordinate, [xWay, yWay]
        # curPos: Current x, y position from world coordinate, [xCur, yCur]
        # relative: If false, it returns new local target point from world coordinate [xTarWld, yTarWld] list,
        # if true, [xTarRel, yTarRel] list.
        isVisible = False
        xRel = waypoint[0] - curPos[0]
        yRel = waypoint[1] - curPos[1]
        rel_pos = np.array([xRel, yRel])
        rel_yaw = np.arctan2(yRel, xRel)
        dist2goal = np.linalg.norm(rel_pos)

        dir_err = self.newCandts[:, 0] - rel_yaw
        idx = 0
        if np.abs(np.amin(dir_err)) <= np.deg2rad(self.yaw2goal_thresh):
            Adir_err = np.abs(dir_err)
            idx = np.argmax(Adir_err)

            if dist >= dist2goal:
                isVisible = True    # When waypoint is directly reachable


        if isVisible:       # Go to the waypoint directly
            xTarRel = dist2goal * np.cos(self.newCandts[idx, 0])
            yTarRel = dist2goal * np.sin(self.newCandts[idx, 0])
            xTarWld = waypoint[0]
            yTarWld = waypoint[1]

        else:
            newDist = np.zeros(len(self.newCandts))
            newPos = np.zeros((len(self.newCandts),2))
            newRelPos = np.zeros((len(self.newCandts),2))
            for i in range(len(self.newCandts)):
                d1 = dist      # Possible moving distance without collision
                x1 = d1 * np.cos(self.newCandts[i, 0])     # New relative position after moving to ith direction and distance
                y1 = d1 * np.sin(self.newCandts[i, 0])
                x1Wld = curPos[0] + x1                      # New absolute position after moving to ith direction and distance
                y1Wld = curPos[1] + y1
                p1Wld = np.array([x1Wld, y1Wld])
                wayWld = np.array([waypoint[0], waypoint[1]])
                d2 = np.linalg.norm(wayWld - p1Wld)         # Distance between New absolute position and waypoint
                totalD = d1 + d2                            # Total moving distance
                newDist[i] = totalD
                newPos[i, 0] = x1Wld
                newPos[i, 1] = y1Wld
                newRelPos[i, 0] = x1
                newRelPos[i, 1] = y1

            shortestIdx = np.argmax(newDist)                # Find shortest moving distance
            xTarWld = newPos[shortestIdx, 0]
            yTarWld = newPos[shortestIdx, 1]
            xTarRel = newRelPos[shortestIdx, 0]
            yTarRel = newRelPos[shortestIdx, 1]


        if not relative:
            return [xTarWld, yTarWld]
        else:
            return [xTarRel, yTarRel]