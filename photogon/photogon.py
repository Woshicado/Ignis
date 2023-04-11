#!/usr/bin/env python3

import numpy as np
import argparse
import os
import json
from datetime import datetime
from pathlib import Path
from utils import load_api
from config import *


def rad_to_deg(rad):
    return rad * 180.0 / np.pi

def deg_to_rad(deg):
    return deg * np.pi / 180.0

def spherical_to_cartesian(theta, phi):
    """
    Transforms given spherical coordinates to cartesian ones on the unit sphere
    """
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # transpose for correct dimension when this function is called with a vector
    return np.array([x, y, z]).transpose()



def theta_from_idx(idx, theta_steps, theta_max):
    """
    Theta cannot be 0 or pi since this would be colinear to our up vector.
    This function handles these cases separately.
    Currently also does not allow theta to be equal to pi / 2 since this is some
    kind of an edge case as well.
    """
    stepsize = theta_max / theta_steps
    if (idx == 0):
        return stepsize * 0.01
    if (theta_steps == idx and (theta_max == np.pi / 2 or theta_max == np.pi)):
        return theta_max - stepsize * 0.01
    return idx * stepsize

def phi_samples_from_theta(theta, thetap_smp):
    """Returns the number of samples for phi for a specific value of theta"""
    pi_2 = np.pi / 2.0
    abs_theta = np.abs(theta - pi_2)    # Center theta s.t. x-y-plane is maximum (pi / 2)

    for perc, smp in thetap_smp:
        if abs_theta > (1-perc) * pi_2:
            num_s = smp

            return num_s

    return thetap_smp[-1][1] # Take last phi steps as fallback



def print_header(file, args, sun_theta, sun_phi, num_samples, additional_headers=""):
    file.write(
            f"#Synthetic BRDF data created in ignis\n"
            f"#created at { datetime.now().strftime('%Y-%m-%d %H:%M:%S') }\n"
            f"#sample_name {Path(args.InputFile).stem}_{sun_theta:.1f}_{sun_phi:.1f}\n"
            f"#datapoints_in_file { num_samples }\n"
            f"#incident_angle\t{rad_to_deg(sun_theta):f}\t{rad_to_deg(sun_phi):f}\n"
            f"#intheta\t{rad_to_deg(sun_theta):f}\n"
            f"#inphi\t{rad_to_deg(sun_phi):f}\n"
            f"#format: theta\tphi\tBRDF\n" + additional_headers
        )

def log(file, args, theta: float, phi: float, brdf: float):
    """Log one line into file in the format: "theta\tphi\tbrdf"""
    out = f"{rad_to_deg(theta):f}\t{rad_to_deg(phi):f}\t{brdf:e}\n"
    file.write(out)

    if args.print:
        print(out, end="")



def num_samples_per_sun(config):
    if config["grid_gen"] == sample_grid:
        return calc_num_samples_per_sun(config)
    else:
        return simulate_num_samples_per_sun(config)

def calc_num_samples_per_sun(config):
    theta_steps = config["theta_steps"]
    theta_max   = config["theta_max"]
    phi_steps   = config["phi_steps"]

    num_samples = 0
    for theta_i in range(0, theta_steps + 1):
        theta = theta_from_idx(theta_i, theta_steps, theta_max)
        num_samples += phi_samples_from_theta(theta, phi_steps)

    return num_samples

def simulate_num_samples_per_sun(config):
    grid_gen = config["grid_gen"]
    acc = 0
    for _theta, _phi, _ori in grid_gen(config):
        acc += 1

    return acc



def sample_grid(config):
    theta_steps = config["theta_steps"]
    theta_max   = config["theta_max"]
    thetap_smp  = config["phi_steps"]
    cam_dist    = config["cam_dist"]
    phi_max     = config["phi_max"]

    original_up = np.array([0, 0, 1])

    for theta_i in range(0, theta_steps + 1):
        theta = theta_from_idx(theta_i, theta_steps, theta_max)
        num_s = phi_samples_from_theta(theta, thetap_smp)
        for phi_i in range(num_s):
            phi   = phi_max / num_s * phi_i # - np.pi
            cart  = spherical_to_cartesian(theta, phi)

            eye   = cam_dist * cart
            dir   = -1 * cart

            right = np.cross(dir, original_up)
            right = right / np.linalg.norm(right)
            up    = np.cross(right, dir)
            up    = up / np.linalg.norm(up)
            ori   = (eye, up, dir)

            yield theta, phi, ori

def replicated_grid(config, path="out/sent_data.txt"):
    """
    Replicates the angles theta and phi from the given phi to sample in the exact
    same pattern.
    Each call yields the next theta, phi, orientation tuple.
    """
    cam_dist    = config["cam_dist"]
    original_up = np.array([0, 0, 1])

    with open(path, "r") as pdata:
        for line in pdata:
            floats = [float(i) for i in line.split()]
            theta = deg_to_rad(floats[0])
            phi   = deg_to_rad(floats[1])

            cart  = spherical_to_cartesian(theta, phi)

            eye   = cam_dist * cart
            dir   = -1 * cart

            right = np.cross(dir, original_up)
            right = right / np.linalg.norm(right)
            up    = np.cross(right, dir)
            up    = up / np.linalg.norm(up)
            ori   = (eye, up, dir)

            yield theta, phi, ori



def render(file, ignis, args, config, sun):
    """
    Render loop to generate the brdf samples defined by config and dumps them
    into file. ignis needs to contain the loaded ignis api.
    """
    grid_generator = config["grid_gen"]

    opts = ignis.RuntimeOptions.makeDefault()
    # opts.OverrideFilmSize = [x, y]

    entity_scene = ignis.Scene().loadFromFile(str(args.InputFile.resolve()),
                                              ignis.SceneParser.F_LoadFilm     |
                                              ignis.SceneParser.F_LoadTechnique|
                                              ignis.SceneParser.F_LoadBSDFs    |
                                              ignis.SceneParser.F_LoadMedia    |
                                              ignis.SceneParser.F_LoadTextures |
                                              ignis.SceneParser.F_LoadShapes   |
                                              ignis.SceneParser.F_LoadEntities |
                                              ignis.SceneParser.F_LoadExternals)
    entity_scene.addFrom(sun)

    with ignis.loadFromScene(entity_scene, opts) as runtime:

        if runtime is None:
            raise RuntimeError("Could not load scene")
        orientation = runtime.InitialCameraOrientation

        for idx, (theta, phi, ori) in enumerate(grid_generator(config)):
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

            if args.store_render:
                img = img.reshape(runtime.FramebufferWidth, runtime.FramebufferHeight, 3)
                ignis.saveExr(os.path.join(config["render_out"], f"{idx:05d}.exr"), img)

            # Clear Framebuffer and reset samplecount for next rotation position
            runtime.reset()

def render_sun(ignis, args, config):
    """
    Loops over all sun positions and creates a brdf sample file for every one.
    Calls the `render` function to fill the brdf sample files correctly.
    """
    theta_steps = config["sun_theta_steps"]
    theta_max   = config["sun_theta_max"]
    theta_stepsize = theta_max / theta_steps

    sun_idx = 0

    # Sun vector can be parallel to scene up vector
    for theta_i in range(0, theta_steps + 1):
        theta = theta_i * theta_stepsize

        # Sun Phi calulation
        thetap_smp = config["sun_phi_steps"]
        phi_max    = config["phi_max"]
        irradiance = config["sun_irradiance"]

        num_s = phi_samples_from_theta(theta, thetap_smp)

        for sphi_i in range(num_s):
            phi = phi_max * (sphi_i / num_s)
            dir = -1 * spherical_to_cartesian(theta, phi)

            # Create Sun scene, as I could not add a sun in any other easy way
            o_json = {}
            o_json["lights"] = [{
                "type": "directional",
                "name": "Sun",
                "direction": dir.tolist(),
                "irradiance": [irradiance, irradiance, irradiance]}]
            json_dump = json.dumps(o_json)

            sunlight = ignis.Scene().loadFromString(json_dump)

            # Manage output paths & start render
            outdir = os.path.join(args.out, str(sun_idx) + ".txt")
            if args.store_render:
                config["render_out"] = os.path.join(args.out, "renders", str(sun_idx))

            os.makedirs(os.path.dirname(outdir), exist_ok=True)
            with open(outdir, "w") as file:
                print_header(file, args, theta, phi, num_samples_per_sun(config))
                render(file, ignis, args, config, sunlight)

            sun_idx += 1



def logtest(args, config):
    """
    For test purposes only!

    Generates a brdf samples file with the angles theta and phi from the defined
    grid generator in the config argument. The brdf value is a constant magic
    number.
    """
    grid_generator = config["grid_gen"]
    outdir = os.path.join(args.out, "0.txt")
    with open(outdir, "w") as file:
        print_header(file, args, 0.0, 0.0, num_samples_per_sun(config))
        for theta, phi, _ in grid_generator(config):
            brdf = 2.456176e-01
            log(file, args, theta, phi, brdf)



def main(ignis, args, config):
    render_sun(ignis, args, config)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('InputFile', type=Path,
                        help='Path to a scene file')
    parser.add_argument('--out', default="out", type=Path,
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
    parser.add_argument('--sun-t-steps', type=int, dest="sun_t_steps",
                        help="Equidistant number of steps for theta")
    parser.add_argument('--sun-t-max', type=float, dest="sun_t_max",
                        help="Maximal value for theta")
    parser.add_argument('--sun-p-max', type=float, dest="sun_p_max",
                        help="Maximal value for phi")
    parser.add_argument('--store-render', action="store_true", dest="store_render",
                        help="Store rendered images in subdirectory 'renders'")
    parser.add_argument('--num-samples', action="store_true", dest="num_samples",
                        help="Print the number of samples to standard out and exit")

    args = parser.parse_args()
    config = standard
    # config["grid_gen"] = sample_grid
    config["grid_gen"] = replicated_grid

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
    if args.sun_t_steps:
        config["sun_theta_steps"] = args.sun_t_steps
    if args.sun_t_max:
        config["sun_theta_max"] = args.sun_t_max
    if args.sun_p_max:
        config["sun_phi_max"] = args.sun_p_max

    if args.num_samples:
        print(f"# Samples for given config: { num_samples_per_sun(config) }")
        exit(0)

    ignis = load_api()
    ignis.setQuiet(args.quiet)

    main(ignis, args, config)
    # logtest(args, config)
