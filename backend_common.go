package mlx

func backendDeviceForGPULayers(gpuLayers int) (device string, partialOffloadUnsupported bool) {
	if gpuLayers == 0 {
		return "cpu", false
	}
	return "gpu", gpuLayers > 0
}
