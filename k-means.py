 #-*- coding:utf8 -*-

import tensorflow as tf

class KMeansCluster:
    def __init__(self, n_clusters, vector_dim):
        self.n_clusters = n_clusters
        self.vector_dim = vector_dim

        with tf.device('/cpu:0'): # use no gpu
            self.session = tf.Session()
            with self.session.as_default()
                self.samples = tf.placeholder(tf.float32, shape=(None, vector_dim))
                self.centroids = tf.placeholder(tf.float32, shape=(n_clusters, vector_dim))

                self.initial_centroids = self.choose_random_centroids(self.samples)

                self.nearest_indices, self.mean_distance = self.assign_to_nearest(self.samples, self.centroids)
                self.updated_centroids = self.update_centroids(self.samples, self.nearest_indices)


    def choose_random_centroids(self, samples):
        n_samples = tf.shape(samples)[0]
        random_indices = tf.random_shuffle(tf.range(0, n_samples))
        centroid_indices = tf.slice(random_indices, [0,], [self.n_clusters,])
        return tf.gather(samples, centroid_indices)


    def assign_to_nearest(self, samples, centroids):
        expanded_vectors = tf.expand_dims(samples, 0)
        expanded_centroids = tf.expand_dims(centroids, 1)
        distances = tf.reduce_sum(tf.square(
            tf.subtract(expanded_vectors, expanded_centroids)), 2)

        nearest_indices = tf.argmin(distances, 0)

        rows = tf.expand_dims(nearest_indices, 1)
        cols = tf.expand_dims(tf.to_int64(tf.range(tf.shape(samples)[0])), 1)
        indices = tf.concat([rows, cols], -1)
        distances_to_centroids = tf.gather_nd(distances, indices)
        mean_distance = tf.reduce_mean(distances_to_centroids)

        return nearest_indices, mean_distance


    def update_centroids(self, samples, nearest_indices):
        nearest_indices = tf.to_int32(nearest_indices)
        partitions = tf.dynamic_partition(samples, nearest_indices, self.n_clusters)
        return tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0)
            for partition in partitions], 0)


    def train(self, samples, max_iter, threshold):
        with self.session.as_default():
            print 'Initializing centroids...'
            last_centroids = self.session.run(
                self.initial_centroids,
                {self.samples : samples})
            last_mean_distance = 1e30

            print 'Start training...'
            for step in xrange(max_iter):
                nearest_indices, centroids, mean_distance = self.session.run(
                        [self.nearest_indices, self.updated_centroids, self.mean_distance],
                        {self.samples : samples, self.centroids: last_centroids})
                print 'step %d, mean distance: %f'%(step, mean_distance)

                if last_mean_distance - mean_distance < threshold:
                    break

                last_centroids = centroids
                last_mean_distance = mean_distance

        return centroids, nearest_indices, mean_distance


    def predict(self, centroids, samples):
        with self.session.as_default():
            nearest_indices = self.session.run(self.nearest_indices,
                    {self.samples: samples, self.centroids: centroids})
            return nearest_indices
