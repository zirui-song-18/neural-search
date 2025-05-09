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
import static org.apache.lucene.util.VectorUtil.add;

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
        long startTimeTotal = System.nanoTime();
        int numClusters = clusterIds.size();
        int maxFeatureDimension = 0;
        long startTimeStep1 = System.nanoTime();
        for (float[] centroid : denseCentroids) {
            maxFeatureDimension = Math.max(maxFeatureDimension, centroid.length);
        }
        long endTimeStep1 = System.nanoTime();
        System.out.println("步骤1 - 找出最大特征维度: " + (endTimeStep1 - startTimeStep1) / 1_000_000.0 + " ms");

        float[][] transposedCenters = new float[maxFeatureDimension][numClusters];
        long startTimeStep2 = System.nanoTime();
        // for (int i = 0; i < clusterIds.size(); i++) {
        // float[] center = denseCentroids.get(clusterIds.get(i));
        // for (int featureIndex = 0; featureIndex < center.length; featureIndex++) {
        // transposedCenters[featureIndex][i] = center[featureIndex];
        // }
        // }
        float[] flatTransposedCenters = new float[maxFeatureDimension * numClusters];
        int size = clusterIds.size();

        for (int i = 0; i < size; i++) {
            float[] center = denseCentroids.get(clusterIds.get(i));
            int centerLength = center.length;

            for (int featureIndex = 0; featureIndex < centerLength; featureIndex++) {
                flatTransposedCenters[featureIndex * numClusters + i] = center[featureIndex];
            }
        }
        long endTimeStep2 = System.nanoTime();
        System.out.println("步骤2 - 转置中心点: " + (endTimeStep2 - startTimeStep2) / 1_000_000.0 + " ms");

        float[] similarities = new float[numClusters];
        long startTimeStep3 = System.nanoTime();
        for (DocFreq docFreq : documents) {
            Arrays.fill(similarities, 0.0f);

            SparseVector docVector = reader.read(docFreq.getDocID());
            if (docVector == null) {
                continue;
            }
            IteratorWrapper<Item> iterator = docVector.iterator();
            float[] scaledFeatureValues = new float[numClusters];
            while (iterator.hasNext()) {
                Item item = iterator.next();
                int tokenIndex = item.getToken();
                float tokenValue = item.getFreq();

                if (tokenIndex < transposedCenters.length) {
                    float[] featureClusterValues = transposedCenters[tokenIndex];

                    System.arraycopy(featureClusterValues, 0, scaledFeatureValues, 0, numClusters);

                    scaleVector(scaledFeatureValues, tokenValue);
                    add(similarities, scaledFeatureValues);
                }
            }

            // 找出最相似的聚类
            int bestCluster = 0;
            for (int i = 1; i < similarities.length; i++) {
                if (similarities[i] > similarities[bestCluster]) {
                    bestCluster = i;
                }
            }

            docAssignments.get(bestCluster).add(docFreq);
        }
        long endTimeStep3 = System.nanoTime();
        System.out.println("步骤3 - 文档分配 (总计): " + (endTimeStep3 - startTimeStep3) / 1_000_000.0 + " ms");

        // 总时间
        long endTimeTotal = System.nanoTime();
        System.out.println("总执行时间: " + (endTimeTotal - startTimeTotal) / 1_000_000.0 + " ms");
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
