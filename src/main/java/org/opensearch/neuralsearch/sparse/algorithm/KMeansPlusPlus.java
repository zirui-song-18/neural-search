/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.sparse.algorithm;

import lombok.AllArgsConstructor;
import org.opensearch.neuralsearch.sparse.common.DocFreq;
import org.opensearch.neuralsearch.sparse.common.SparseVector;
import org.opensearch.neuralsearch.sparse.common.SparseVectorReader;
import org.opensearch.neuralsearch.sparse.common.IteratorWrapper;
import org.opensearch.neuralsearch.sparse.common.SparseVector.Item;

import java.io.IOException;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * KMeans++ clustering algorithm
 */
@AllArgsConstructor
public class KMeansPlusPlus implements Clustering {

    private final static int MINIMAL_CLUSTER_DOC_SIZE = 10;

    private final float alpha;
    private final int beta;
    private final SparseVectorReader reader;

    /**
     * Assigns documents to clusters based on similarity.
     *
     * @param documents The list of documents to assign
     * @param docAssignments The list of document assignments for each cluster
     * @param denseCentroids The list of cluster centroids
     * @param clusterIds The list of cluster IDs to consider
     */
    private void assignDocumentsToCluster(
        List<DocFreq> documents,
        List<List<DocFreq>> docAssignments,
        List<float[]> denseCentroids,
        List<Integer> clusterIds
    ) {
        int numClusters = clusterIds.size();
        int maxFeatureDimension = 0;
        for (int clusterId : clusterIds) {
            maxFeatureDimension = Math.max(maxFeatureDimension, denseCentroids.get(clusterId).length);
        }

        float[][] transposedCenters = new float[maxFeatureDimension][numClusters];

        int clusterIndex = 0;
        for (int clusterId : clusterIds) {
            float[] center = denseCentroids.get(clusterId);
            for (int featureIndex = 0; featureIndex < center.length; featureIndex++) {
                transposedCenters[featureIndex][clusterIndex] = center[featureIndex];
            }
            clusterIndex++;
        }
        float[] similarities = new float[numClusters];

        Arrays.fill(similarities, 0.0f);

        for (DocFreq docFreq : documents) {
            SparseVector docVector = reader.read(docFreq.getDocID());
            if (docVector == null) {
                continue;
            }
            IteratorWrapper<Item> iterator = docVector.iterator();
            while (iterator.hasNext()) {
                Item item = iterator.next();
                int tokenIndex = item.getToken();
                float tokenValue = item.getFreq();

                // 确保索引在范围内
                if (tokenIndex < transposedCenters.length) {
                    float[] featureClusterValues = transposedCenters[tokenIndex];

                    // 对每个聚类，累加该特征的贡献
                    for (int j = 0; j < Math.min(numClusters, featureClusterValues.length); j++) {
                        similarities[j] += tokenValue * featureClusterValues[j];
                    }
                }
            }
            int bestCluster = 0;
            for (int i = 1; i < similarities.length; i++) {
                if (similarities[i] > similarities[bestCluster]) {
                    bestCluster = i;
                }
            }
            // float maxScore = Float.MIN_VALUE;
            //
            // for (int clusterId : clusterIds) {
            // float[] center = denseCentroids.get(clusterId);
            // if (center != null) {
            // float score = docVector.dotProduct(center);
            // if (score > maxScore) {
            // maxScore = score;
            // bestCluster = clusterId;
            // }
            // }
            // }

            docAssignments.get(bestCluster).add(docFreq);
        }
    }

    @Override
    public List<DocumentCluster> cluster(List<DocFreq> docFreqs) throws IOException {
        if (beta == 1) {
            DocumentCluster cluster = new DocumentCluster(null, docFreqs, true);
            return List.of(cluster);
        }
        int size = docFreqs.size();

        // Avoid number of clusters too large
        int num_cluster = Math.min(beta, size / MINIMAL_CLUSTER_DOC_SIZE);

        // Generate beta unique random centers
        Random random = new Random();
        int[] centers = random.ints(0, size).distinct().limit(num_cluster).toArray();

        // Initialize centroids
        List<List<DocFreq>> docAssignments = new ArrayList<>(num_cluster);
        List<float[]> denseCentroids = new ArrayList<>();
        for (int i = 0; i < num_cluster; i++) {
            docAssignments.add(new ArrayList<>());
            SparseVector center = reader.read(docFreqs.get(centers[i]).getDocID());
            if (center == null) {
                denseCentroids.add(null);
            } else {
                denseCentroids.add(center.toDenseVector());
            }
        }

        // Create a list of all cluster indices
        List<Integer> allClusterIds = IntStream.range(0, num_cluster).boxed().collect(Collectors.toList());

        // Assign documents to clusters
        assignDocumentsToCluster(docFreqs, docAssignments, denseCentroids, allClusterIds);

        // Identify valid clusters
        List<Integer> validClusterIds = IntStream.range(0, num_cluster)
            .filter(i -> docAssignments.get(i).size() >= MINIMAL_CLUSTER_DOC_SIZE)
            .boxed()
            .collect(Collectors.toList());

        // Only proceed with reassignment if we have valid clusters to reassign to
        if (!validClusterIds.isEmpty()) {
            // Identify small clusters and collect their documents for reassignment using streams
            List<DocFreq> docsToReassign = IntStream.range(0, num_cluster)
                .filter(i -> docAssignments.get(i).size() < MINIMAL_CLUSTER_DOC_SIZE)
                .mapToObj(i -> {
                    List<DocFreq> docs = new ArrayList<>(docAssignments.get(i));
                    docAssignments.get(i).clear();
                    return docs;
                })
                .flatMap(List::stream)
                .collect(Collectors.toList());

            // Reassign documents from small clusters
            assignDocumentsToCluster(docsToReassign, docAssignments, denseCentroids, validClusterIds);
        }

        List<DocumentCluster> clusters = new ArrayList<>();
        for (int i = 0; i < num_cluster; ++i) {
            if (docAssignments.get(i).isEmpty()) {
                continue;
            }
            DocumentCluster cluster = new DocumentCluster(null, docAssignments.get(i), false);
            PostingsProcessor.summarize(cluster, this.reader, this.alpha);
            clusters.add(cluster);
        }
        return clusters;
    }
}
