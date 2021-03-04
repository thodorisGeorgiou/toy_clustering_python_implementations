import numpy
import time
import multiprocessing
import sklearn.metrics.pairwise

def updateCenter(data):
	sums = numpy.zeros([data[1], data[0][0].shape[1]])
	numExamples = []
	for c in range(data[1]):
		examples = numpy.where(data[0][1]==c)
		numExamples.append(examples[0].shape[0])
		if numExamples[-1] > 0:
			sums[c] = numpy.sum(data[0][0][examples], axis=0)
	return [sums, numpy.array(numExamples)]

def calculateDataLabel(data):
	return numpy.argmin(data)

def calculateDataLabelApproximate(data):
	labels = []
	for point in range(data[0].shape[0]):
		labels.append(numpy.argmin(data[0][point, data[1]]))
	return numpy.array(labels)

def assignCluster(data):
	return numpy.where(data[0]==data[1])

def calculateCenter(cluster):
	return numpy.median(cluster, axis=0)


class KMedians:
	"""!
	@brief Class represents clustering algorithm K-Means.
	
	"""
	
	def __init__(self, n_clusters, initial_centers=None, tolerance=0.001, maxIter=300, nJobs=1):
		"""!
		@brief Constructor of clustering algorithm K-Means.
		
		"""
		self.data = None
		self.centers = initial_centers
		self.n_clusters = n_clusters
		self.labels = []
		self.clusters = []
		self.distances = None
		self.maxIter = maxIter
		self.nJobs = nJobs
		self.Pool = multiprocessing.Pool(nJobs)

	def newUniqueCenter(self):
		c = numpy.random.randint(self.data.shape[0])
		center = self.data[c]
		if numpy.any(numpy.all(center==self.centers, axis=1)):
			center = self.newUniqueCenter()
		return center

	def updateCenters(self):
		st = time.time()
		_centers = self.Pool.map(calculateCenter, [self.data[c] for c in self.clusters])
		self.Pool.close()
		self.Pool.join()
		self.Pool = multiprocessing.Pool(self.nJobs)
		for ic, c in enumerate(_centers):
			if numpy.any(numpy.isnan(c)):
				nc = self.newUniqueCenter()
				_centers[ic] = nc
		self.centers = numpy.array(_centers)
		print(time.time() - st, end=" ")
		unique, counts = numpy.unique(self.centers, axis=0, return_counts=True)
		nonUnique = numpy.where(counts>1)
		for i in nonUnique[0]:
			for j in range(counts[i]-1):
				c = self.newUniqueCenter()
				demporoallo = numpy.where(numpy.all(unique[i]==self.centers, axis=1))[0][0]
				self.centers[demporoallo] = c

		print(time.time() - st, end=" ")
		# st = time.time()
		# centerDists = 
		# zeros = numpy.where(centerDists==0)
		# for i in range(zeros[0].shape[0]):
		# 	if zeros[0][i] != zeros[1][i]:
		# 		c = self.newUniqueCenterDistanceBased()
		# 		self.centerIndexes[zeros[1][i]] = c
		# 		self.centers[zeros[1][i]] = self.data[c]
		# 		centerDists = self.distances[self.centerIndexes, :][:,self.centerIndexes]
		# 		zeros = numpy.where(centerDists==0)
		# print(time.time() - st, end=" ")
		# print()

	def calculateLabels(self, data=None):
		distances = sklearn.metrics.pairwise_distances(self.data, self.centers, metric="hamming", n_jobs=self.nJobs)
		_labels = self.Pool.map(calculateDataLabel, distances)
		labels = numpy.array(_labels)
		self.Pool.close()
		self.Pool.join()
		self.Pool = multiprocessing.Pool(self.nJobs)
		# numLabels = numpy.unique(labels)
		# if numLabels.shape[0] != self.n_clusters:
		# 	print(numpy.unique(self.centers, axis=0).shape)
		# 	print("LALALALALALALALALALALALAS")
		# 	exit()
		return labels

	def fit(self, data):
		if data.dtype != numpy.uint8:
			exit("KMedians Error: Wrong data type")
		if len(data.shape) != 2:
			exit("KMedians Error: Wrong data shape")
		if data.shape[0] < self.n_clusters:
			print("Number of clusters can't be larger than number of data points")
			exit()
			return None
		self.data = numpy.unpackbits(data, axis=1)
		if self.centers is None:
			centerIndexes = numpy.random.choice(data.shape[0], size=self.n_clusters, replace=False)
			self.centers = self.data[centerIndexes]
		unique, counts = numpy.unique(self.centers, axis=0, return_counts=True)
		nonUnique = numpy.where(counts>1)
		for i in nonUnique[0]:
			for j in range(counts[i]-1):
				c = self.newUniqueCenter()
				demporoallo = numpy.where(numpy.all(unique[i]==self.centers, axis=1))[0][0]
				self.centers[demporoallo] = c
			print("resolving center duplicates")
		if self.data.shape[1] != self.centers.shape[1]:
			print("KMedians Error: Not correct number of features")
			exit()
			return None
		# self.distances = numpy.array_split(sklearn.metrics.pairwise_distances(data, metric="euclidean", n_jobs=self.nJobs), self.nJobs, axis=0)
		for i in range(self.maxIter):
			print(i)
			oldCenters = numpy.copy(self.centers)
			print("Predict", end=": ")
			st = time.time()
			self.labels = self.calculateLabels(data)
			print(time.time()-st)
			print("Assign", end=": ")
			st = time.time()
			self.clusters = self.Pool.map(assignCluster, [[self.labels, c] for c in range(self.n_clusters)])
			self.Pool.close()
			self.Pool.join()
			self.Pool = multiprocessing.Pool(self.nJobs)
			print(time.time()-st)
			# exit()
			print("Fit", end=": ")
			st = time.time()
			self.updateCenters()
			print(time.time()-st)
			if numpy.all(self.centers==oldCenters): break
		del self.data
		del self.distances
		self.Pool.close()
		self.Pool.join()
		self.Pool = None
		return self

	def predict(self, data):
		_data = numpy.unpackbits(data, axis=1)
		if self.centers is None:
			print("KMedians Error: No centers available")
			return
		if len(_data.shape) == 1 and _data.shape[0] == self.centers.shape[1]:
			_data = numpy.expand_dims(_data, axis=0)
		elif len(_data.shape) == 2 and _data.shape[1] == self.centers.shape[1]:
			_data = _data
		else:
			print(_data.shape)
			print("KMedians Error: Not correct input for prediction")
			exit()
			return
		dists = sklearn.metrics.pairwise_distances(_data, self.centers, metric="hamming", n_jobs=self.nJobs)
		labels = numpy.argmin(dists, axis=1)
		return labels