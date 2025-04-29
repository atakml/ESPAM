package lts.pattern;

/**This holds the node ids for a particular embedding of a Pattern.*/
public abstract class EmbeddingEdge {

	public abstract EmbeddingEdge getParent();
	public abstract EdgeType getType();

	/** Returns an array of the node ids in the embedding.
	 *  If a corresponding PatternEdge is traced, the int pairs from it mean that node ids at those indices
	 *  in the array have an edge between them.
	 */
	public int[] traceNodes() {
		int[] embeddingNodeIds = new int[countNodes()];
		traceNodes(embeddingNodeIds);
		return embeddingNodeIds;
	}
	
	/** Returns an array of the node ids in the embedding.
	 *  If a corresponding PatternEdge is traced, the int pairs from it mean that node ids at those indices
	 *  in the array have an edge between them.
	 */
	public void traceNodes(int[] embeddingNodeIdsHolderArray) {
		int i = 0;
		for (EmbeddingEdge ee = this; ee != null; ee = ee.getParent()) {
			switch (ee.getType()) {
			case BASE:
				embeddingNodeIdsHolderArray[i++] = ((EmbeddingEdgeBase) ee).getNodeId1();
				embeddingNodeIdsHolderArray[i++] = ((EmbeddingEdgeBase) ee).getNodeId2();
				break;
			case EXISTING_NODES:
				break;	
			case NEW_NODE:
				embeddingNodeIdsHolderArray[i++] = ((EmbeddingEdgeToNewNode) ee).getNodeId();
				break;
			}
		}
	}

	public int countNodes() {
		int c = 0;
		for (EmbeddingEdge ee = this; ee != null; ee = ee.getParent()) {
			switch (ee.getType()) {
			case BASE: 
				c += 2;
				break;
			case EXISTING_NODES:
				break;
			case NEW_NODE: 
				c++;
				break;
			}
		}
		return c;
	}
}
