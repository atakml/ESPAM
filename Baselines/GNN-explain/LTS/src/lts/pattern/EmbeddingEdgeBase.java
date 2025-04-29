package lts.pattern;

/**The base edge (first edge) in an embedding.*/
public class EmbeddingEdgeBase extends EmbeddingEdge {
	
	private final int nodeId1, node2Id2;

	public EmbeddingEdgeBase(int node1Id, int node2Id) {
		this.nodeId1 = node1Id;
		this.node2Id2 = node2Id;
	}

	public int getNodeId1() {
		return nodeId1;
	}

	public int getNodeId2() {
		return node2Id2;
	}

	@Override
	public EmbeddingEdge getParent() {
		return null;
	}

	@Override
	public EdgeType getType() {
		return EdgeType.BASE;
	}	
	
}
