#!/usr/bin/env python3

import numpy as np
import argparse
import os
import sys
from matplotlib import pyplot as plt
from typing import Tuple
from pathlib import Path
from utils import load_api
from config import *


def rad_to_deg(rad: float) -> float:
    return rad * 180.0 / np.pi


def spherical_to_cartesian(theta: float, phi: float) -> Tuple[float, float, float]:
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return x, y, z



def render_theta(file, ignis, theta: float, args, config, callnum) -> float:
    thetap_smp = config["phi_steps"]
    camera_distance = config["cam_dist"]
    phi_max = config["phi_max"]

    pi_2 = np.pi / 2.0
    abs_theta = np.abs(theta - pi_2)    # Center theta s.t. x-y-plane is maximum (pi / 2)
    up_   = np.array([0, 0, 1])

    num_s = thetap_smp[-1][1]           # Take last phi steps as fallback
    for perc, smp in thetap_smp:
        if abs_theta > perc * pi_2:
            num_s = smp
            break

    for phi_i in range(num_s):
        phi   = phi_max * (phi_i / num_s)
        cart  = spherical_to_cartesian(theta, phi)

        eye   = camera_distance * np.array(cart)
        dir   = -1 * np.array(cart)

        right = np.cross(dir, up_)
        right = right / np.linalg.norm(right)
        up    = np.cross(right, dir)
        up    = up / np.linalg.norm(up)
        ori   = np.array((eye, up, dir))

        render(file, ignis, args, config, ori, theta, phi, callnum + phi_i)

    return phi_i


def render(file, ignis, args, config, ori, theta, phi, callnum):
    opts = ignis.RuntimeOptions.makeDefault()
    # opts.OverrideFilmSize = [x, y]

    with ignis.loadFromFile(str(args.InputFile.resolve()), opts) as runtime:

        if runtime is None:
            raise RuntimeError("Could not load scene")

        orientation = runtime.InitialCameraOrientation
        orientation.Eye = ori[0]
        orientation.Up  = ori[1]
        orientation.Dir = ori[2]
        runtime.setCameraOrientationParameter(orientation)

        while runtime.SampleCount < config["spp"]:
            runtime.step()

        img = np.divide(runtime.getFramebuffer(), runtime.IterationCount)\
                .reshape(runtime.FramebufferHeight, runtime.FramebufferWidth, 3)

        mid_horizontal, mid_vertical = runtime.FramebufferWidth // 2, runtime.FramebufferHeight // 2
        brdf = img[mid_vertical, mid_horizontal, 0]

        log(file, args, theta, phi, brdf)

        if args.render_out:
            img = img.reshape(runtime.FramebufferWidth, runtime.FramebufferHeight, 3)
            ignis.saveExr(os.path.join(args.render_out, f"{callnum:03d}.exr"), img)


def log(file, args, theta: float, phi: float, brdf: float):
    out = f"{ rad_to_deg(theta):f}\t{ rad_to_deg(phi):f}\t{ brdf:g}\n"
    file.write(out)

    if args.print:
        print(out, end="")


def main(ignis, args, config):
    theta_steps = config["theta_steps"]
    theta_max   = config["theta_max"]

    theta_stepsize = theta_max / theta_steps

    with open(args.out, "w") as file:
        file.write(FILE_HEADER)

        callnum = 0
        init_theta = theta_stepsize * 0.01

        callnum += render_theta(file, ignis, init_theta, args, config, callnum)

        for theta_i in range(1, theta_steps):
            theta = theta_i * theta_stepsize
            callnum += render_theta(file, ignis, theta, args, config, callnum)

        theta = theta_max
        if theta_max == np.pi:
            theta = theta - init_theta

        callnum += render_theta(file, ignis, theta, args, config, callnum)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('InputFile', type=Path,
                        help='Path to a scene file')
    parser.add_argument('--out', default="out.txt", type=Path,
                        help='Path to export brdf values to')
    parser.add_argument('--spp', type=int,
                        help="Number of samples to acquire for each direction")
    parser.add_argument('--quiet', action="store_true",
                        help="Make logging quiet")
    parser.add_argument('--print', action="store_true",
                        help="Print scanned BRDF values to standard out")
    parser.add_argument('--cam-dist', type=int, dest="cam_dist",
                        help="Camera Distance from (0,0,0)")
    parser.add_argument('--t-steps', type=int, dest="t_steps",
                        help="Equidistant number of steps for theta")
    parser.add_argument('--t-max', type=float, dest="t_max",
                        help="Maximal value for theta")
    parser.add_argument('--p-max', type=float, dest="p_max",
                        help="Maximal value for phi")
    parser.add_argument('--render-out', default="", dest="render_out",
                        help="Directory to store rendered images. Won't be stored if not given")

    args = parser.parse_args()
    config = standard

    if args.cam_dist:
        config["cam_dist"] = args.cam_dist
    if args.t_steps:
        config["theta_steps"] = args.theta_steps
    if args.t_max:
        config["theta_max"] = args.t_max
    if args.p_max:
        config["phi_max"] = args.p_max
    if args.spp:
        config["spp"] = args.spp

    ignis = load_api()
    ignis.setQuiet(args.quiet)

    main(ignis, args, config)
