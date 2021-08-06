import camera_model as cam_model

pinhole_rot_noise_10k_od = cam_model.PinholeModelRotNoiseLearning10kRayoRayd
pinhole_rot_noise_10k_od_dist = cam_model.PinholeModelRotNoiseLearning10kRayoRaydDistortion

camera_dict = {
    "pinhole_rot_noise_10k_rayo_rayd": pinhole_rot_noise_10k_od,
    "pinhole_rot_noise_10k_rayo_rayd_dist": pinhole_rot_noise_10k_od_dist
}

