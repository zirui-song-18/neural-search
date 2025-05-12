/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.opensearch.neuralsearch.sparse.algorithm;

import lombok.AllArgsConstructor;
import org.opensearch.neuralsearch.sparse.common.DocFreq;
import org.opensearch.neuralsearch.sparse.common.SparseVector;
import org.opensearch.neuralsearch.sparse.common.SparseVectorReader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.apache.lucene.util.Constants;

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
    // private void assignDocumentsToCluster(
    // List<DocFreq> documents,
    // List<List<DocFreq>> docAssignments,
    // List<float[]> denseCentroids,
    // List<Integer> clusterIds
    // ) {
    // // long startTimeTotal = System.nanoTime();
    // int numClusters = clusterIds.size();
    // int maxFeatureDimension = 0;
    // // long startTimeStep1 = System.nanoTime();
    // for (float[] centroid : denseCentroids) {
    // maxFeatureDimension = Math.max(maxFeatureDimension, centroid.length);
    // }
    // // long endTimeStep1 = System.nanoTime();
    // // System.out.println("步骤1 - 找出最大特征维度: " + (endTimeStep1 - startTimeStep1) / 1_000_000.0 + " ms");
    //
    // float[][] transposedCenters = new float[maxFeatureDimension][numClusters];
    // // long startTimeStep2 = System.nanoTime();
    // for (int i = 0; i < clusterIds.size(); i++) {
    // float[] center = denseCentroids.get(clusterIds.get(i));
    // for (int featureIndex = 0; featureIndex < center.length; featureIndex++) {
    // transposedCenters[featureIndex][i] = center[featureIndex];
    // }
    // }
    // // long endTimeStep2 = System.nanoTime();
    // // System.out.println("步骤2 - 转置中心点: " + (endTimeStep2 - startTimeStep2) / 1_000_000.0 + " ms");
    //
    // float[] similarities = new float[numClusters];
    // // long startTimeStep3 = System.nanoTime();
    // float[] constntArray = new float[numClusters];
    // float result = 0;
    // float[] resultArray = new float[numClusters];
    // for (DocFreq docFreq : documents) {
    // Arrays.fill(similarities, 0.0f);
    //
    // SparseVector docVector = reader.read(docFreq.getDocID());
    // if (docVector == null) {
    // continue;
    // }
    // IteratorWrapper<Item> iterator = docVector.iterator();
    // float[] scaledFeatureValues = new float[numClusters];
    // while (iterator.hasNext()) {
    // Item item = iterator.next();
    // int tokenIndex = item.getToken();
    // float tokenValue = item.getFreq();
    //
    // if (tokenIndex < transposedCenters.length) {
    //
    // Arrays.fill(constntArray, tokenValue);
    // result = dotProduct(constntArray, transposedCenters[tokenIndex]);
    // Arrays.fill(resultArray, result);
    // add(similarities, resultArray);
    // }
    // }
    //
    // // 找出最相似的聚类
    // int bestCluster = 0;
    // for (int i = 1; i < similarities.length; i++) {
    // if (similarities[i] > similarities[bestCluster]) {
    // bestCluster = i;
    // }
    // }
    //
    // docAssignments.get(bestCluster).add(docFreq);
    // }
    // // long endTimeStep3 = System.nanoTime();
    // // System.out.println("步骤3 - 文档分配 (总计): " + (endTimeStep3 - startTimeStep3) / 1_000_000.0 + " ms");
    //
    // // 总时间
    // // long endTimeTotal = System.nanoTime();
    // // System.out.println("总执行时间: " + (endTimeTotal - startTimeTotal) / 1_000_000.0 + " ms");
    // }
    // private void assignDocumentsToCluster(
    // List<DocFreq> documents,
    // List<List<DocFreq>> docAssignments,
    // List<float[]> denseCentroids,
    // List<Integer> clusterIds
    // ) {
    // int numClusters = clusterIds.size();
    // int maxFeatureDimension = 0;
    // for (float[] centroid : denseCentroids) {
    // maxFeatureDimension = Math.max(maxFeatureDimension, centroid.length);
    // }
    //
    // // Transpose centroids for better memory access patterns
    // float[][] transposedCenters = new float[maxFeatureDimension][numClusters];
    // for (int i = 0; i < clusterIds.size(); i++) {
    // float[] center = denseCentroids.get(clusterIds.get(i));
    // for (int featureIndex = 0; featureIndex < center.length; featureIndex++) {
    // transposedCenters[featureIndex][i] = center[featureIndex];
    // }
    // }
    //
    // // Process documents in parallel for large datasets
    // documents.forEach(docFreq -> {
    // float[] similarities = new float[numClusters];
    //
    // SparseVector docVector = reader.read(docFreq.getDocID());
    // if (docVector == null) {
    // return; // Skip this document
    // }
    //
    // // Process each token in the document
    // IteratorWrapper<Item> iterator = docVector.iterator();
    // while (iterator.hasNext()) {
    // Item item = iterator.next();
    // int tokenIndex = item.getToken();
    // float tokenValue = item.getFreq();
    //
    // // Skip if token index is out of bounds
    // if (tokenIndex >= transposedCenters.length) {
    // continue;
    // }
    //
    // // Direct vector scaling and accumulation
    // float[] featureClusterValues = transposedCenters[tokenIndex];
    // scaleAndAccumulate(similarities, featureClusterValues, tokenValue, numClusters);
    // }
    //
    // // Find best cluster (most similar)
    // int bestCluster = findMaxIndex(similarities);
    //
    // // Thread-safe add to cluster assignments
    // synchronized (docAssignments.get(bestCluster)) {
    // docAssignments.get(bestCluster).add(docFreq);
    // }
    // });
    // }

    private void assignDocumentsToCluster(
        List<DocFreq> documents,
        List<List<DocFreq>> docAssignments,
        List<float[]> denseCentroids,
        List<Integer> clusterIds
    ) {

        for (DocFreq docFreq : documents) {
            SparseVector docVector = reader.read(docFreq.getDocID());
            if (docVector == null) {
                continue;
            }

            int bestCluster = 0;
            float maxScore = Float.MIN_VALUE;

            for (int clusterId : clusterIds) {
                float[] center = denseCentroids.get(clusterId);
                if (center != null) {
                    float score = docVector.dotProduct(center);
                    if (score > maxScore) {
                        maxScore = score;
                        bestCluster = clusterId;
                    }
                }
            }

            docAssignments.get(bestCluster).add(docFreq);
        }
    }

    /**
     * Optimized method to scale vector values by a constant and accumulate into result
     * Uses loop unrolling and FMA operations when available
     */
    private void scaleAndAccumulate(float[] result, float[] vector, float scale, int length) {
        int i = 0;

        // Use unrolled loop for better performance on larger vectors
        if (length > 16) {
            int limit = length & ~(8 - 1); // Round down to multiple of 8
            for (; i < limit; i += 8) {
                result[i] = fma(vector[i], scale, result[i]);
                result[i + 1] = fma(vector[i + 1], scale, result[i + 1]);
                result[i + 2] = fma(vector[i + 2], scale, result[i + 2]);
                result[i + 3] = fma(vector[i + 3], scale, result[i + 3]);
                result[i + 4] = fma(vector[i + 4], scale, result[i + 4]);
                result[i + 5] = fma(vector[i + 5], scale, result[i + 5]);
                result[i + 6] = fma(vector[i + 6], scale, result[i + 6]);
                result[i + 7] = fma(vector[i + 7], scale, result[i + 7]);
            }
        }

        // Handle remaining elements
        for (; i < length; i++) {
            result[i] = fma(vector[i], scale, result[i]);
        }
    }

    /**
     * Find the index of the maximum value in an array
     */
    private int findMaxIndex(float[] array) {
        int maxIndex = 0;
        float maxValue = array[0];

        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    /**
     * FMA (Fused Multiply-Add) operation: a*b + c
     * Uses hardware acceleration if available
     */
    private static float fma(float a, float b, float c) {
        if (Constants.HAS_FAST_SCALAR_FMA) {
            // System.out.println("Accelerated");
            return Math.fma(a, b, c);
        } else {
            // System.out.println("Not Accelerated");
            return a * b + c;
        }
    }

    private float[] scaleVector(float[] vector, float scalar) {
        // 使用循环展开技术
        int i = 0;
        int length = vector.length;

        // 每次处理4个元素
        for (; i <= length - 8; i += 8) {
            vector[i] *= scalar;
            vector[i + 1] *= scalar;
            vector[i + 2] *= scalar;
            vector[i + 3] *= scalar;
            vector[i + 4] *= scalar;
            vector[i + 5] *= scalar;
            vector[i + 6] *= scalar;
            vector[i + 7] *= scalar;
        }

        // 处理剩余元素
        for (; i < length; i++) {
            vector[i] *= scalar;
        }

        return vector;  // 返回同一个数组的引用
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
