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
	labels = []
	for p in data[0]:
		labels.append(numpy.argmin(numpy.linalg.norm(p-data[1], axis=1)))
	return numpy.array(labels)

def calculateDataLabelApproximate(data):
	labels = []
	for point in range(data[0].shape[0]):
		labels.append(numpy.argmin(data[0][point, data[1]]))
	return numpy.array(labels)

def assignCluster(data):
	return numpy.where(data[0]==data[1])

def calculateCenter(cluster):
	return numpy.mean(cluster, axis=0)

class KMeans:
	"""!
	@brief Class represents clustering algorithm K-Means.
	
	"""
	
	def __init__(self, n_clusters, initial_centers=None, tolerance=0.001, maxIter=3000, nJobs=1):
		"""!
		@brief Constructor of clustering algorithm K-Means.
		
		"""
		# _data = numpy.array(data)
		# self.data = numpy.array(data)
		self.splitted_data = []
		self.centers = initial_centers
		self.centerIndexes = None
		self.distances = None
		self.n_clusters = n_clusters
		self.labels = []
		self.tolerance = tolerance
		self.maxIter = maxIter
		self.nJobs = nJobs
		self.Pool = multiprocessing.Pool(nJobs)
		self.emptyClusters = None

	def updateCenters(self):
		mapInput = [[d, self.n_clusters] for d in zip(self.splitted_data, numpy.array_split(self.labels, self.nJobs, axis=0))]
		partCenters = self.Pool.map(updateCenter, mapInput)
		self.Pool.close()
		self.Pool.join()
		self.Pool = multiprocessing.Pool(self.nJobs)
		self.centers = numpy.sum([pc[0] for pc in partCenters], axis=0)
		population = numpy.expand_dims(numpy.sum([pc[1] for pc in partCenters], axis=0), axis=1)
		self.emptyClusters = numpy.where(population==0)
		population[self.emptyClusters] = 1
		self.centers /= population

	def calculateLabels(self, data=None):
		if data is None:
			split_data = self.splitted_data
		else:
			split_data = numpy.array_split(data, self.nJobs, axis=0)
		# _distances = sklearn.metrics.pairwise_distances(data, self.centers, metric="euclidean", n_jobs=self.nJobs)
		# labels = numpy.array(self.Pool.map(calculateDataLabel, _distances))
		labels = numpy.concatenate(self.Pool.map(calculateDataLabel, [[d, self.centers] for d in split_data]), axis=0)
		self.Pool.close()
		self.Pool.join()
		self.Pool = multiprocessing.Pool(self.nJobs)
		return labels

	def fit(self, data):
		if len(data.shape) != 2:
			exit("KMeans Error: Wrong data shape")
		if data.shape[0] < self.n_clusters:
			print("Number of clusters can't be larger than number of data points")
			exit()
			return None
		if self.centers is None:
			self.centers = data[numpy.random.choice(data.shape[0], size=self.n_clusters, replace=False)]
		if data.shape[1] != self.centers.shape[1]:
			print("KMeans Error: Wrong number of features")
			exit()
			return None
		self.splitted_data = numpy.array_split(data, self.nJobs, axis=0)
		for i in range(self.maxIter):
			oldCenters = numpy.copy(self.centers)
			print("Predict")
			if not self.emptyClusters is None:
				self.centers[self.emptyClusters[0]] = data[numpy.random.choice(data.shape[0], size=self.emptyClusters[0].shape[0], replace=False)]
				self.emptyClusters = None
			self.labels = self.calculateLabels(data)
			print("Fit")
			self.updateCenters()
			if numpy.all(self.centers==oldCenters): break
		del self.splitted_data

	def predict(self, data):
		if self.centers==None:
			print("KMeans Error: No centers available")
			return
		if data.shape[1] != self.centers.shape[1]:
			print("KMeans Error: Not correct number of features")
			return
		labels = self.calculateLabels(data)
		return labels

class ApproximateKMeans:
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
		if c in self.centerIndexes:
			c = self.newUniqueCenter()
		return c

	def newUniqueCenterDistanceBased(self):
		c = numpy.random.randint(self.data.shape[0])
		if numpy.any(self.distances[c, self.centerIndexes]==0):
			c = self.newUniqueCenterDistanceBased()
		return c

	def updateCenters(self):
		st = time.time()
		self.centers = numpy.array(self.Pool.map(calculateCenter, [self.data[c] for c in self.clusters]))
		self.Pool.close()
		self.Pool.join()
		self.Pool = multiprocessing.Pool(self.nJobs)
		print(time.time() - st, end=" ")
		try:
			st = time.time()
			_distances = sklearn.metrics.pairwise_distances(self.centers, self.data, metric="euclidean", n_jobs=self.nJobs)
			print(time.time() - st, end=" ")
		except ValueError:
			for center in self.centers:
				if numpy.any(numpy.isnan(center)) or numpy.any(numpy.isinf(center)):
					print(center)
			exit()
		st = time.time()
		self.centerIndexes = numpy.argmin(_distances, axis=1)
		print(time.time() - st, end=" ")
		st = time.time()
		unique, counts = numpy.unique(self.centerIndexes, return_counts=True)
		nonUnique = numpy.where(counts>1)
		for i in nonUnique[0]:
			# print(nonUnique)
			# print("Substituting "+str(unique[i])+" "+str(counts[i]-1)+" times, with ", end="")
			for j in range(counts[i]-1):
				c = self.newUniqueCenter()
				# print(c, end=" ")
				demporoallo = numpy.where(self.centerIndexes==unique[i])[0][0]
				self.centerIndexes[demporoallo] = c
				self.centers[demporoallo] = self.data[c]
		# centerDists = sklearn.metrics.pairwise_distances(self.centers, metric="euclidean", n_jobs=self.nJobs)
		print(time.time() - st, end=" ")
		st = time.time()
		centerDists = self.distances[self.centerIndexes, :][:,self.centerIndexes]
		zeros = numpy.where(centerDists==0)
		for i in range(zeros[0].shape[0]):
			if zeros[0][i] != zeros[1][i]:
				c = self.newUniqueCenterDistanceBased()
				self.centerIndexes[zeros[1][i]] = c
				self.centers[zeros[1][i]] = self.data[c]
				centerDists = self.distances[self.centerIndexes, :][:,self.centerIndexes]
				zeros = numpy.where(centerDists==0)
		print(time.time() - st, end=" ")
		# print()

	def calculateLabels(self, data=None):
		# labels = numpy.concatenate(self.Pool.map(calculateDataLabelApproximate, [[d, self.centerIndexes] for d in self.distances]), axis=0)
		labels = []
		# counter = 0
		for point in range(self.distances.shape[0]):
			labels.append(numpy.argmin(self.distances[point, self.centerIndexes]))
			# if point in self.centerIndexes:
			# 	if labels[-1] != numpy.where(self.centerIndexes==point)[0][0]:
			# 		print(str(point)+": "\
			# 		+str(self.distances[point, point])+" - "\
			# 		+str(numpy.min(self.distances[point, self.centerIndexes]))+" - "\
			# 		+str(labels[-1])+" - "\
			# 		+str(numpy.where(self.centerIndexes==point)[0])+" - "\
			# 		+str(numpy.argmin(self.distances[point, self.centerIndexes]))+" - "\
			# 		+str(self.centerIndexes[labels[-1]]))
			# 		print(self.data[point]==self.data[self.centerIndexes[labels[-1]]])
				# counter += 1
		labels = numpy.array(labels)
		self.Pool.close()
		self.Pool.join()
		self.Pool = multiprocessing.Pool(self.nJobs)
		numLabels = numpy.unique(labels)
		if numLabels.shape[0] != self.n_clusters:
			print(numpy.unique(self.centerIndexes).shape)
			print("LALALALALALALALALALALALAS")
			exit()
		return labels

	def fit(self, data):
		if len(data.shape) != 2:
			exit("KMeans Error: Wrong data shape")
		if data.shape[0] < self.n_clusters:
			print("Number of clusters can't be larger than number of data points")
			exit()
			return None
		if self.centers is None:
			self.centerIndexes = numpy.random.choice(data.shape[0], size=self.n_clusters, replace=False)
			self.centers = data[self.centerIndexes]
		if data.shape[1] != self.centers.shape[1]:
			print("KMeans Error: Not correct number of features")
			exit()
			return None
		self.data = data
		self.distances = sklearn.metrics.pairwise_distances(data, metric="euclidean", n_jobs=self.nJobs)
		centerDists = self.distances[self.centerIndexes, :][:,self.centerIndexes]
		print(centerDists.shape)
		zeros = numpy.where(centerDists==0)
		print(len(zeros))
		print(zeros[0].shape)
		print(zeros[1].shape)
		for i in range(zeros[0].shape[0]):
			if zeros[0][i] != zeros[1][i]:
				c = self.newUniqueCenterDistanceBased()
				self.centerIndexes[zeros[1][i]] = c
				self.centers[zeros[1][i]] = self.data[c]
				centerDists = self.distances[self.centerIndexes, :][:,self.centerIndexes]
				zeros = numpy.where(centerDists==0)
				if zeros[0].shape[0] == self.n_clusters: break
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
		if self.centers is None:
			print("KMeans Error: No centers available")
			return
		if len(data.shape) == 1 and data.shape[0] == self.centers.shape[1]:
			_data = numpy.expand_dims(data, axis=0)
		elif len(data.shape) == 2 and data.shape[1] == self.centers.shape[1]:
			_data = data
		else:
			print(data.shape)
			print("KMeans Error: Not correct input for prediction")
			exit()
			return
		labels = numpy.argmin(sklearn.metrics.pairwise_distances(_data, self.centers, metric="euclidean", n_jobs=self.nJobs), axis=1)
		return labels