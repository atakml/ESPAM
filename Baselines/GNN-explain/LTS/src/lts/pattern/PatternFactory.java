package lts.pattern;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import lts.graph.GraphsContext;
import util.collections.ArraysUtil;
import util.collections.CollectionUtil;

/**Factory class to construct Patterns. You can add EmbeddingEdges to this.*/
public class PatternFactory {

	private final PatternEdge patternEdge;
	private final HashMap<Integer, ArrayList<EmbeddingEdge>> graphIdToEmbeddingsMap = new HashMap<>();
	
	private int minGraphId = Integer.MAX_VALUE;
	
	public PatternFactory(PatternEdge patternEdge) {
		this.patternEdge = patternEdge;
	}

	public void addEmbedding(int graphId, EmbeddingEdge embedding) {
		minGraphId = Math.min(minGraphId, graphId);  // track the minimum graph id, for CCAM code generation
		CollectionUtil.addItemToMapOfArrayLists(graphIdToEmbeddingsMap, graphId, embedding);
	}
	
	public boolean hasAPositiveEmbedding(GraphsContext graphsContext) {
		return minGraphId < graphsContext.getNumPositiveGraphs();  
	}
	
	public CCAMCode generateCCAMCode(GraphsContext graphsContext) {
		return new CCAMCode(patternEdge, findMinEmbedding(), graphsContext.getGraph(minGraphId));
	}
	
	// for ccam code
	private EmbeddingEdge findMinEmbedding() {
		
		List<EmbeddingEdge> embeddingsOfMinGraph = graphIdToEmbeddingsMap.get(minGraphId);
		
		if (embeddingsOfMinGraph.size() == 1) {  // only one embedding, so it must be the min 
			return embeddingsOfMinGraph.get(0);
		}
		
		// get smallest node ids from each embedding, record in minNodeIds
		// the embedding with the smallest smallest is the min Embedding
		int[] minNodeIds = new int[embeddingsOfMinGraph.size()];
		int idx = 0;
		for (EmbeddingEdge embedding : embeddingsOfMinGraph) {
			minNodeIds[idx++] = getMinNodeId(embedding);
		}

		int[] indicesOfEmbeddingsTiedForSmallestSmallestNodeId = ArraysUtil.getMinIndices(minNodeIds);
		if (indicesOfEmbeddingsTiedForSmallestSmallestNodeId.length == 1) {
			return embeddingsOfMinGraph.get(indicesOfEmbeddingsTiedForSmallestSmallestNodeId[0]);
		}
		
		// at least 2 embeddings tied for smallest smallest node id
		// we need to compare all the nodes now
		int currMinEmbeddingId = indicesOfEmbeddingsTiedForSmallestSmallestNodeId[0];
		int[] currMinEmbeddingSortedNodes = embeddingsOfMinGraph.get(currMinEmbeddingId).traceNodes();
		
		Arrays.sort(currMinEmbeddingSortedNodes);
		
		int[] nextEmbeddingSortedNodes = new int[currMinEmbeddingSortedNodes.length];
		for (int i = 1; i < indicesOfEmbeddingsTiedForSmallestSmallestNodeId.length; i++) {
			
			EmbeddingEdge nextEmbedding = embeddingsOfMinGraph.get(indicesOfEmbeddingsTiedForSmallestSmallestNodeId[i]);
			nextEmbedding.traceNodes(nextEmbeddingSortedNodes);
			
			Arrays.sort(nextEmbeddingSortedNodes);
			
			if (ArraysUtil.LEX_COMPARATOR_INT_ARRAYS.compare(currMinEmbeddingSortedNodes, nextEmbeddingSortedNodes) > 0) {
				currMinEmbeddingId = indicesOfEmbeddingsTiedForSmallestSmallestNodeId[i];
				int[] swapTmp = currMinEmbeddingSortedNodes;
				currMinEmbeddingSortedNodes = nextEmbeddingSortedNodes;
				nextEmbeddingSortedNodes = swapTmp;
			}
		}
		
		return embeddingsOfMinGraph.get(currMinEmbeddingId);
	}
	
	private int getMinNodeId(EmbeddingEdge embedding) {
		
		int minNodeId = Integer.MAX_VALUE;
		int tmp = -1;
		
		for (; embedding != null; embedding = embedding.getParent()) {
			switch (embedding.getType()) {
			case BASE:
				if ((tmp = ((EmbeddingEdgeBase) embedding).getNodeId1()) < minNodeId) minNodeId = tmp;
				if ((tmp = ((EmbeddingEdgeBase) embedding).getNodeId2()) < minNodeId) minNodeId = tmp;
				break;
			case EXISTING_NODES:
				break;
			case NEW_NODE:
				if ((tmp = ((EmbeddingEdgeToNewNode) embedding).getNodeId()) < minNodeId) minNodeId = tmp;
				break;
			}
		}

		return minNodeId;
	}
	
	public Pattern generatePattern() {
		return new Pattern(patternEdge, graphIdToEmbeddingsMap);
	}
}
