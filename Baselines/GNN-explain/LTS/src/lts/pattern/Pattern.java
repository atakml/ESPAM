package lts.pattern;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import lts.graph.Graph;
import lts.graph.GraphsContext;
import lts.graph.NeighborNodeIdAndEdgeLabel;
import lts.main.input_preprocessing.IntToStringDecoder;
import util.collections.CompactIntListMap;
import util.collections.IntArrayView;
import util.collections.IntListMapEntry;

/**Represents a subgraph, i.e. a pattern.
 * This is basically a pattern edge which defines the structure of the pattern
 * and a multimap of graph ids to embeddings of this pattern.
 * The state of a Pattern is not to be changed after construction from a PatternFactory.
 */
public class Pattern {
	
	// pattern information
	private final PatternEdge pattern;
	
	// Multimap of graph ids to embeddings of this pattern.
	// Using a custom class instead of the standard Hash or TreeMap to save memory,
	// since A LOT of instances of this class will be created.
	// Does not and cannot have empty entries, if the key (graph id) is in this,
	// there is at least 1 embedding in that graph.
	private final CompactIntListMap<EmbeddingEdge> graphIdsToEmbeddings;
	
	/**Constructs a pattern given a PatternEdge and a multimap of graph ids to embeddings of the pattern.
	 * The state of a Pattern is not to be changed after construction from a PatternFactory.
	 */
	protected <L extends List<EmbeddingEdge>> Pattern(PatternEdge pattern, HashMap<Integer, L> embeddingsMap) {
		this.pattern = pattern;
		this.graphIdsToEmbeddings = new CompactIntListMap<>(embeddingsMap, false);
	}

	/**Returns how many positive graphs have at least 1 embedding of this pattern.*/
	public int getNumPositiveEmbeddings(GraphsContext graphsContext) {
		return binarySearchNumberOfPositiveEmbeddings(graphsContext.getNumPositiveGraphs());
	}
	
	/**Returns how many negative graphs have at least 1 embedding of this pattern.*/
	public int getNumNegativeEmbeddings(GraphsContext graphsContext) {
		return getNumEmbeddings() - getNumPositiveEmbeddings(graphsContext);
	}

	private int binarySearchNumberOfPositiveEmbeddings(int positiveGraphs) {
		IntArrayView array = graphIdsToEmbeddings.getKeys();
		int start = 0;
		int end = array.length();
		while (start < end) {
			int mid = start + (end - start) / 2;
			int currMidNodeId = array.get(mid);
			if (positiveGraphs < currMidNodeId) {
				end = mid;
			} else if (currMidNodeId < positiveGraphs) {
				start = mid + 1;
			} else {
				return mid;
			}
		}
		return start;
	}

	/**Returns to total number of graphs with at least 1 embedding of this pattern.*/
	public int getNumEmbeddings() {
		return graphIdsToEmbeddings.getKeys().length();
	}

	/**Returns sorted list of graph ids that have this pattern embedded.*/
	public IntArrayView getEmbeddingGraphIds() {
		return graphIdsToEmbeddings.getKeys();
	}
	
	// Pairs the 2 indices of an edge in the pattern
	private static class PatternEdgeNodeIndices {
		
		final int nodeIndex1, nodeIndex2;

		PatternEdgeNodeIndices(int nodeIndex1, int nodeIndex2) {
			this.nodeIndex1 = nodeIndex1;
			this.nodeIndex2 = nodeIndex2;
		}

		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + nodeIndex1;
			result = prime * result + nodeIndex2;
			return result;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			PatternEdgeNodeIndices other = (PatternEdgeNodeIndices) obj;
			if (nodeIndex1 != other.nodeIndex1)
				return false;
			if (nodeIndex2 != other.nodeIndex2)
				return false;
			return true;
		}
		
	}
	
	/**Returns a collection of factories that can generate the child patterns of this pattern.*/
	public Collection<PatternFactory> generateChildPatternFactories(GraphsContext graphsContext) {
		
		HashSet<PatternEdgeNodeIndices> patternEdgesSet = new HashSet<>();
		int[] patternEdges = pattern.traceEdges();
		for (int i = 0; i < patternEdges.length; i += 2) {
			PatternEdgeNodeIndices patternEdgeNodeIndices =
					new PatternEdgeNodeIndices(patternEdges[i], patternEdges[i + 1]);
			patternEdgesSet.add(patternEdgeNodeIndices);
		}
		
		int[] embeddingNodesArray = new int[graphIdsToEmbeddings.getFirst().countNodes()];
		PatternExtensionEmbeddingAccumulator extensionAccumulator = new PatternExtensionEmbeddingAccumulator(pattern);
		
		for (IntListMapEntry<EmbeddingEdge> embeddingsForSingleGraph : graphIdsToEmbeddings) {
			int graphId = embeddingsForSingleGraph.getKey();
			for (EmbeddingEdge embedding : embeddingsForSingleGraph.getValue()) {
				addExtensionProposalsOfSingleEmbedding(
						graphId, 
						graphsContext,
						embedding,
						embeddingNodesArray,
						patternEdgesSet,
						extensionAccumulator);
			}
		}
		
		return extensionAccumulator.getPatternFactories();
	}
	
	private void addExtensionProposalsOfSingleEmbedding(
			int graphId,
			GraphsContext graphsContext,
			EmbeddingEdge embedding,
			int[] embeddingNodesArray,
			HashSet<PatternEdgeNodeIndices> patternEdgesSet,
			PatternExtensionEmbeddingAccumulator accumulator) {

		Graph graph = graphsContext.getGraph(graphId);
		
		embedding.traceNodes(embeddingNodesArray);

		HashMap<Integer, Integer> embeddingNodeIdsToPatternIndexMap = new HashMap<>();
		for (int i = 0; i < embeddingNodesArray.length; i++) {
			embeddingNodeIdsToPatternIndexMap.put(embeddingNodesArray[i], i);
		}

		for (Map.Entry<Integer, Integer> embeddingNodeIdsToPatternIndex : embeddingNodeIdsToPatternIndexMap.entrySet()) {

			int embeddingNodeId = embeddingNodeIdsToPatternIndex.getKey();
			int patternNodeIndex = embeddingNodeIdsToPatternIndex.getValue();

			for (NeighborNodeIdAndEdgeLabel nodeNeighborIdAndEdgeLabel : graph.getNeighbors(embeddingNodeId)) {

				int neighborEmbeddingNodeId = nodeNeighborIdAndEdgeLabel.getToNodeId();
				Integer neighborPatternNodeIndex = embeddingNodeIdsToPatternIndexMap.get(neighborEmbeddingNodeId);

				if (neighborPatternNodeIndex == null) {
					// add Existing Node to New Node proposal if the node to extend to is not already in the pattern
				
					PatternExtension extension = new PatternExtensionToNewNode(
							patternNodeIndex,
							graph.getNodeLabel(nodeNeighborIdAndEdgeLabel.getToNodeId()),
							nodeNeighborIdAndEdgeLabel.getEdgeLabel());

					accumulator.addProposal(extension, graphId, new EmbeddingEdgeToNewNode(embedding, neighborEmbeddingNodeId));

				} else if (patternNodeIndex < neighborPatternNodeIndex &&
						!patternEdgesSet.contains(new PatternEdgeNodeIndices(patternNodeIndex, neighborPatternNodeIndex))) {
					// add Existing Node to Existing Node proposal if the edge does not already exist
					// consider only proposals where the extending node has a smaller pattern ID, to prevent duplicates

					PatternExtension extension = new PatternExtensionBetweenExistingNodes(
							patternNodeIndex,
							neighborPatternNodeIndex,
							nodeNeighborIdAndEdgeLabel.getEdgeLabel());

					accumulator.addProposal(extension, graphId, new EmbeddingEdgeBetweenExistingNodes(embedding));			
				}
			}
		}
	}
	
	private static class PatternIndexAndEdgeLabel {
		int patternIndex, edgeLabel;
	}
	
	/**Output all the information about this pattern as a String.
	 * @param decoder Decoder to translate the int ids and labels back into their original Strings.
	 * @return
	 */
	public String toString(GraphsContext graphsContext, IntToStringDecoder decoder) {
		
		int[] embeddingNodes = graphIdsToEmbeddings.getFirst().traceNodes();
		int[] patternEdges = pattern.traceEdges();
		int graphId = graphIdsToEmbeddings.getKeys().get(0);
		Graph graph = graphsContext.getGraph(graphId);
		
		int[] nodeLabels = new int[embeddingNodes.length];
		for (int i = 0; i < embeddingNodes.length; i++) {
			nodeLabels[i] = graph.getNodeLabel(embeddingNodes[i]);
		}
		
		ArrayList<ArrayList<PatternIndexAndEdgeLabel>> patternIndexToNeighbors = new ArrayList<>(embeddingNodes.length);
		for (int i = 0; i < embeddingNodes.length; i++) {
			patternIndexToNeighbors.add(new ArrayList<PatternIndexAndEdgeLabel>());
		}
		
		for (int i = 0; i < patternEdges.length; i += 2) {
			int patternIndex1 = patternEdges[i];
			int patternIndex2 = patternEdges[i + 1];
			
			int edgeLabel = graph.getEdgeLabel(embeddingNodes[patternIndex1], embeddingNodes[patternIndex2]);
			
			for (int j = 0; j < 2; j++) {
				PatternIndexAndEdgeLabel neighbor = new PatternIndexAndEdgeLabel();
				neighbor.patternIndex = j == 0 ? patternIndex2 : patternIndex1;
				neighbor.edgeLabel = edgeLabel;
				patternIndexToNeighbors.get(j == 0 ? patternIndex1 : patternIndex2).add(neighbor);
			}
		}
		
		StringBuilder sb = new StringBuilder();
		for (int patternIndex = 0; patternIndex < patternIndexToNeighbors.size(); patternIndex++) {
			sb.append(String.format("%d[%s] :", patternIndex, decoder.decodeNodeLabel(nodeLabels[patternIndex])));
			for (PatternIndexAndEdgeLabel patternIndexAndEdgeLabel : patternIndexToNeighbors.get(patternIndex)) {
				sb.append(' ').append(patternIndexAndEdgeLabel.patternIndex);
				sb.append('(').append(decoder.decodeEdgeLabel(patternIndexAndEdgeLabel.edgeLabel)).append(')');
			}
			sb.append('\n');
		}
		
		sb.append('\n');

		sb.append("Positive Embeddings:\n");

		int positives = getNumPositiveEmbeddings(graphsContext);
		int i = 0;
		
		for (IntListMapEntry<EmbeddingEdge> entry : graphIdsToEmbeddings) {
			int currGraphId = entry.getKey();

			for (EmbeddingEdge embedding : entry.getValue()) {
				sb.append(String.format("%s :", decoder.decodeGraphId(currGraphId)));
				embedding.traceNodes(embeddingNodes);
				for (int node : embeddingNodes) {
					sb.append(' ').append(decoder.decodeNodeIdForGraphs(currGraphId, node));
				}
				sb.append('\n');
			}
			
			if (++i == positives) {
				sb.append('\n');
				sb.append("Negative Embeddings:\n");
			}
		}
		sb.append('\n');

		sb.append(String.format("Positive Support: %d/%d%n", 
				this.getNumPositiveEmbeddings(graphsContext), graphsContext.getNumPositiveGraphs()));
		sb.append(String.format("Negative Support: %d/%d%n", 
				this.getNumNegativeEmbeddings(graphsContext), graphsContext.getNumNegativeGraphs()));
		
		return sb.toString();
	};
	
}
