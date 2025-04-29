package lts.pattern;

import java.util.Arrays;

import lts.graph.Graph;
import lts.main.input_preprocessing.IntToStringDecoder;
import util.collections.BitUtil;

/**Represents CCAM code. Although the inner representation is not as described in the paper,
 * it should function the same.
 * Applicable for patterns with at least 2 nodes. Usable in hashmaps and hashsets.
 */
public class CCAMCode {

	private static class NodeIdAndPatternIndex implements Comparable<NodeIdAndPatternIndex> {
		int nodeId, patternIndex;
		@Override
		public int compareTo(NodeIdAndPatternIndex o) {
			return this.nodeId - o.nodeId;
		}
	}
	
	private static class EdgeMatrixOffsetAndEdgeLabel implements Comparable<EdgeMatrixOffsetAndEdgeLabel> {
		int edgeMatrixOffset, edgeLabel;
		@Override
		public int compareTo(EdgeMatrixOffsetAndEdgeLabel o) {
			return this.edgeMatrixOffset - o.edgeMatrixOffset;
		}
	}
	
	private final int numNodes;
	
	/** Encoded as 
	 * [bits of lower triangular edge existence matrix in left->right, then top->down order, compressed to ints]
	 * [node labels]
	 * [edge labels in same order as corresponding bits in the matrix] */
	private final int[] code;
	
	/**Creates a CCAM code
	 * @param graph Graph the minEmbedding is located in
	 */
	protected CCAMCode(PatternEdge pattern, EmbeddingEdge minEmbedding, Graph graph) {
		
		int[] patternEdges = pattern.traceEdges();
		int[] embeddingNodes = minEmbedding.traceNodes();
		
		NodeIdAndPatternIndex[] sortedNodeIdsAndPatternIndices = new NodeIdAndPatternIndex[embeddingNodes.length];
		for (int i = 0; i < embeddingNodes.length; i++) {
			NodeIdAndPatternIndex nodeIdAndPatternIndex = new NodeIdAndPatternIndex();
			nodeIdAndPatternIndex.nodeId = embeddingNodes[i];
			nodeIdAndPatternIndex.patternIndex = i;
			sortedNodeIdsAndPatternIndices[i] = nodeIdAndPatternIndex;
		}
		
		// sort by node id
		Arrays.sort(sortedNodeIdsAndPatternIndices);
		
		int[] patternIndexToCCAMIndexMap = new int[sortedNodeIdsAndPatternIndices.length];
		for (int i = 0; i < sortedNodeIdsAndPatternIndices.length; i++) {
			patternIndexToCCAMIndexMap[sortedNodeIdsAndPatternIndices[i].patternIndex] = i;
		}
		
		numNodes = embeddingNodes.length;
		
		// calculate where the node labels start by calculate the greatest possible bit positive in the edge matrix
		int nodeAndEdgePtr = calculateBitPosition(numNodes - 2, numNodes - 1) / 32 + 1;
		
		// allocate code array
		code = new int[nodeAndEdgePtr + numNodes + patternEdges.length / 2];
		
		// write node labels
		for (int i = 0; i < numNodes; i++) {
			code[nodeAndEdgePtr++] = graph.getNodeLabel(sortedNodeIdsAndPatternIndices[i].nodeId);
		}
		
		EdgeMatrixOffsetAndEdgeLabel[] sortedMatrixOffsetsAndEdgeLabels = new EdgeMatrixOffsetAndEdgeLabel[patternEdges.length / 2];
		
		// write edge matrix
		for (int i = 0; i < patternEdges.length; i += 2) {
			int patternIndex1 = patternEdges[i];
			int patternIndex2 = patternEdges[i + 1];
			
			int ccamIndex1 = patternIndexToCCAMIndexMap[patternIndex1];
			int ccamIndex2 = patternIndexToCCAMIndexMap[patternIndex2];
			
			if (ccamIndex1 > ccamIndex2) {
				int swapTmp = ccamIndex1;
				ccamIndex1 = ccamIndex2;
				ccamIndex2 = swapTmp;
			}
			
			int offset = calculateBitPosition(ccamIndex1, ccamIndex2);
			
			BitUtil.setIntArrayBit(code, offset, true);
			
			int edgeLabel = graph.getEdgeLabel(embeddingNodes[patternIndex1], embeddingNodes[patternIndex2]);
			EdgeMatrixOffsetAndEdgeLabel matrixOffsetAndEdgeLabel = new EdgeMatrixOffsetAndEdgeLabel();
			matrixOffsetAndEdgeLabel.edgeMatrixOffset = offset;
			matrixOffsetAndEdgeLabel.edgeLabel = edgeLabel;
			sortedMatrixOffsetsAndEdgeLabels[i / 2] = matrixOffsetAndEdgeLabel;
		}
		
		// sort by bit offset of the edge in the flattened triangular edge matrix
		Arrays.sort(sortedMatrixOffsetsAndEdgeLabels);
		
		// write edge labels
		for (int i = 0; i < sortedMatrixOffsetsAndEdgeLabels.length; i++) {
			code[nodeAndEdgePtr++] = sortedMatrixOffsetsAndEdgeLabels[i].edgeLabel;
		}
		
	}
	
	// calculate the offset of the bit representing an edge between ccam indices a and b
	// a should < b
	private int calculateBitPosition(int a, int b) {
		return b * (b - 1) / 2 + a;
	}
	
	private int calculateLabelStartOffset() {
		return calculateBitPosition(numNodes - 2, numNodes - 1) / 32 + 1;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + Arrays.hashCode(code);
		result = prime * result + numNodes;
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
		CCAMCode other = (CCAMCode) obj;
		if (!Arrays.equals(code, other.code))
			return false;
		if (numNodes != other.numNodes)
			return false;
		return true;
	}
	
	// returns nicely formated adjacency matrix and edge labels
	public String toString(IntToStringDecoder decoder) {
		StringBuilder sb = new StringBuilder("Adjaceny Matrix:\n");
		int codePtr = calculateLabelStartOffset();
		for (int i = 0; i < numNodes; i++) {
			for (int j = 0; j < i; j++) {
				boolean hasEdge = BitUtil.getIntArrayBit(code, calculateBitPosition(j, i));
				sb.append(hasEdge ? "1 " : "0 ");
			}
			sb.append('[').append(decoder.decodeNodeLabel(code[codePtr++])).append("]\n");
		}
		sb.append("Edges:\n");
		while (codePtr < code.length) {
			sb.append(decoder.decodeEdgeLabel(code[codePtr++])).append('\n');
		}
		return sb.toString();
	}
	
}
