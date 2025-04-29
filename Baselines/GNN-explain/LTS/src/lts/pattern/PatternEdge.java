package lts.pattern;

/**This describes the structure of a subgraph (Pattern). The structure is just information about edges.
 * It contains no information about node ids or labels of any sort.*/
public abstract class PatternEdge {
	
	public abstract PatternEdge getParent();
	public abstract EdgeType getType();

	/** Returns an array where consecutive pairs of ints
	 *  means that an edge exists between the nodes with the those indices.
	 *  If a corresponding EmbeddingEdge is traced, the indices are that of the node array it returns.
	 */
	public int[] traceEdges() {
		int[] patternEdgeIndices = new int[countEdges() * 2];
		traceEdges(patternEdgeIndices);
		return patternEdgeIndices;
	}
	
	/** Returns an array where consecutive pairs of ints
	 *  means that an edge exists between the nodes with the those indices.
	 *  If a corresponding EmbeddingEdge is traced, the indices are that of the node array it returns.
	 */
	public void traceEdges(int[] patternEdgeHolderArray) {
		int currPatternNodeIndex = 0;
		int i = 0;
		for (PatternEdge pe = this; pe != null; pe = pe.getParent()) {
			switch (pe.getType()) {
			case BASE:
				patternEdgeHolderArray[i++] = currPatternNodeIndex++;
				patternEdgeHolderArray[i++] = currPatternNodeIndex++;
				break;
			case EXISTING_NODES:
				patternEdgeHolderArray[i++] = currPatternNodeIndex + ((PatternEdgeBetweenExistingNodes) pe).getFirstExistingNodeIndex();
				patternEdgeHolderArray[i++] = currPatternNodeIndex + ((PatternEdgeBetweenExistingNodes) pe).getSecondExistingNodeIndex();
				break;
			case NEW_NODE:
				patternEdgeHolderArray[i++] = currPatternNodeIndex++;
				patternEdgeHolderArray[i++] = currPatternNodeIndex + ((PatternEdgeToNewNode) pe).getExistingNodeIndex();
			}
		}
	}

	/** How many edges in the pattern. */
	public int countEdges() {
		int c = 0;
		for (PatternEdge pe = this; pe != null; pe = pe.getParent()) {
			c++;
		}
		return c;
	}

}
