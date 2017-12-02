 #-*- coding:utf8 -*-

import tensorflow as tf

class KMeansCluster:
    def __init__(self, n_clusters, vector_dim):
        self.n_clusters = n_clusters
        self.vector_dim = vector_dim

        self.session = tf.Session()

        with self.session.as_default(), tf.device('/cpu:0'): # use no gpu
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

        nearest_centroids = tf.gather(centroids, nearest_indices)
        distances_to_centroids = tf.square(tf.subtract(nearest_centroids, samples))
        distances_to_centroids = tf.reduce_sum(distances_to_centroids, 1)
        mean_distance = tf.reduce_mean(distances_to_centroids)

        return nearest_indices, mean_distance


    def update_centroids(self, samples, nearest_indices):
        nearest_indices = tf.to_int32(nearest_indices)
        partitions = tf.dynamic_partition(samples, nearest_indices, self.n_clusters)
        return tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0)
            for partition in partitions], 0)


    def train(self, samples, max_iter, threshold, final_centroids_file):
        with self.session.as_default():
            print 'Initializing centroids...'
            last_centroids = self.session.run(
                self.initial_centroids,
                {self.samples : samples})
            last_mean_distance = 1e30

            print 'Start training...'
            for step in xrange(max_iter):
                centroids, mean_distance = self.session.run(
                        [self.updated_centroids, self.mean_distance],
                        {self.samples : samples, self.centroids: last_centroids})
                print 'step %d, mean distance: %f'%(step, mean_distance)


                if last_mean_distance - mean_distance < threshold:
                    break
                last_centroids = centroids
                last_mean_distance = mean_distance

            print 'Outputing the final centroids...'
            with open(final_centroids_file, 'w') as fout:
                for centroid in last_centroids:
                    fout.write(' '.join(map(str, centroid)))
                    fout.write('\n')

        return last_centroids, last_mean_distance


    def load_centroids(self, centroids_file):
        centroids = []
        for line in open(centroids_file):
            centroids.append(map(float, line.strip().split()))
        return centroids


    def predict(self, centroids, word2vec, test_file, output_file):
        with self.session.as_default(), open(output_file, 'w') as fout:
            def run_and_output(samples, lens):
                if len(samples)==0:
                    return
                nearest_indices = self.session.run(self.nearest_indices,
                        {self.samples: samples, self.centroids: centroids})
                idx=0
                for l in lens:
                    fout.write(' '.join(map(str, nearest_indices[idx:idx+l]))+'\n')
                    idx+=l
            count=0
            samples_flat=[]
            lens=[]
            for line in open(test_file):
                samples = map(lambda x:word2vec.get(x, word2vec['</s>']),
                    line.strip().split())
                samples_flat.extend(samples)
                l=len(samples)
                lens.append(l)
                count += l
                if count>=80000:
                    run_and_output(samples_flat, lens)
                    samples_flat=[]
                    lens=[]
                    count=0

            run_and_output(samples_flat, lens)


    def centroids_words(self, text_files, centroids_files, output_file):
        cwords=[{} for _ in xrange(self.n_clusters)]
        for ftext, fcen in zip(text_files, centroids_files):
            for text, cen in zip(open(ftext), open(fcen)):
                words = text.strip().split()
                cens = map(int, cen.strip().split())
                for w, c in zip(words, cens):
                    cwords[c][w]=True
        with open(output_file, 'w') as fout:
            for words in cwords:
                fout.write('%s\n'%(' '.join(words)))
