package lts.pattern;

public class EmbeddingEdgeToNewNode extends EmbeddingEdge {

	private final EmbeddingEdge parent;
	private final int nodeId;
	
	public EmbeddingEdgeToNewNode(EmbeddingEdge parent, int nodeId) {
		this.parent = parent;
		this.nodeId = nodeId;
	}

	@Override
	public EmbeddingEdge getParent() {
		return parent;
	}

	public int getNodeId() {
		return nodeId;
	}

	@Override
	public EdgeType getType() {
		return EdgeType.NEW_NODE;
	}
	
}
