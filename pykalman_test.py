#!/usr/bin/python
import cv2
import math
import numpy
import pykalman


# System to be estimated. Estimating a target position from observations taken from several range sensors.
# target position : p = [p.x, p.y]
# sensor positions: s0 = [s0.x, s0.y], ..., sn = [sn.x, sn.y]
# state space     : x = p
# observation     : y = [|p - s0|, |p - s1|, ..., |p - sn|]
class System:
	# constructor
	def __init__(self):
		# true target and sensor positions
		self.target = numpy.float32((256, 256))
		self.sensors = numpy.float32([(128, 128), (384, 384), (384, 128)])

		cv2.namedWindow('canvas')
		cv2.setMouseCallback('canvas', self.mouse_callback)

	# generates an observation vector
	# [|p - s0|, |p - s1|, ..., |p - sn|] + noise
	def measurement(self):
		dists = numpy.linalg.norm(self.sensors - self.target, axis=1)
		return dists + numpy.abs(numpy.random.normal(0.0, 25.0, dists.size))

	# updates the target position
	def mouse_callback(self, event, x, y, flags, userdata):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.pushed = True
		elif event == cv2.EVENT_LBUTTONUP:
			self.pushed = False

		if hasattr(self, 'pushed') and self.pushed:
			self.target = numpy.float32((x, y))

	# draw
	def draw(self, filter, measurement):
		canvas = numpy.ones((512, 512, 3), dtype=numpy.uint8) * 255

		# draw sensors and observations
		for r, sensor in zip(measurement, self.sensors):
			cv2.circle(canvas, tuple(sensor.astype(numpy.int32)), 5, (0, 255, 0), 2)
			cv2.circle(canvas, tuple(sensor.astype(numpy.int32)), int(r), (128, 128, 128), 1)

		# draw true target position
		cv2.circle(canvas, tuple(self.target.astype(numpy.int32)), 2, (255, 0, 0), 2)

		# draw estimated target position and error ellipse
		cv2.circle(canvas, tuple(filter.mean.astype(numpy.int32)), 5, (0, 0, 255), 2)
		eigenvalues, eigenvectors = numpy.linalg.eig(filter.cov)
		idx = [x[0] for x in sorted(enumerate(eigenvalues), key=lambda x: x[1])]
		angle = math.degrees(math.atan2(eigenvectors[0, idx[1]], eigenvectors[1, idx[1]]))
		axes = (int(eigenvalues[idx[1]]), int(eigenvalues[idx[0]]))
		cv2.ellipse(canvas, tuple(filter.mean.astype(numpy.int32)), axes, angle, 0, 360, (0, 0, 255))

		cv2.imshow('canvas', canvas)


# Target position estimator using UKF
class Filter:
	def __init__(self, sensors):
		self.sensors = sensors

		# initialize ukf
		trans_cov = numpy.eye(2) * 20
		obs_cov = numpy.eye(sensors.shape[0]) * 100
		self.mean = numpy.random.normal(256, 256, 2)
		self.cov = numpy.eye(2) * 128
		self.ukf = pykalman.UnscentedKalmanFilter(self.transition, self.observation, trans_cov, obs_cov, self.mean, self.cov)

	# transition function (adding noise)
	def transition(self, state, noise):
		return state + noise

	# observation function (same as System)
	def observation(self, state, noise):
		expected = numpy.linalg.norm(self.sensors - state, axis=-1)
		return expected + noise

	# update state using ukf
	def update(self, measurement):
		self.mean, self.cov = self.ukf.filter_update(self.mean, self.cov, measurement)


# entry point
def main():
	system = System()
	filter = Filter(system.sensors)

	while cv2.waitKey(50) != 0x1b:
		obs = system.measurement()
		filter.update(obs)
		system.draw(filter, obs)


if __name__ == '__main__':
	main()
