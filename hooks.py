

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.basic_session_run_hooks import _as_graph_element

import numpy as np

import time

class EarlyStopping(session_run_hook.SessionRunHook):

	def __init__(self, metric, start_time, target=0.97, check_every=100, max_secs=10):
		self.metric = metric
		self.target = target
		self.counter = 0
		self.check_every = check_every
		self.max_secs = max_secs
		self.start_time = start_time
		
	def before_run(self, run_context):
		self.counter += 1
		self.should_check = (self.counter % self.check_every) == 0

		if self.should_check:
			return session_run_hook.SessionRunArgs([self.metric])

	def after_run(self, run_context, run_values):
		if self.should_check and run_values.results is not None:
			t = run_values.results[0][1]
			if t > self.target:
				tf.logging.info("Early stopping")
				run_context.request_stop()

		tf.logging.info(f"EarlyStopping time {time.time() - self.start_time} > {self.max_secs}")
			
		if (time.time() - self.start_time) > self.max_secs:
			tf.logging.info("Early stopping")
			run_context.request_stop()
		


class CallbackHook(session_run_hook.SessionRunHook):
	def __init__(self, metrics, callback_after=None, callback_end=None):
		self.metrics = metrics
		self.callback_after = callback_after
		self.callback_end = callback_end

	def before_run(self, run_context):
		return session_run_hook.SessionRunArgs(self.metrics)

	def after_run(self, run_context, run_values):
		if self.callback_after is not None:
			self.callback_after(run_context, run_values)

	def end(self, session):
		if self.callback_end is not None:
			self.callback_end(session)


class MetricHook(session_run_hook.SessionRunHook):
	def __init__(self, metric, cb):
		self.metric = metric
		self.cb = cb
		self.readings = []

	def before_run(self, run_context):
		return session_run_hook.SessionRunArgs([self.metric])

	def after_run(self, run_context, run_values):
		self.readings.append(run_values.results[0][1])

	def end(self, session):
		self.cb(np.average(self.readings))
		self.readings.clear()
