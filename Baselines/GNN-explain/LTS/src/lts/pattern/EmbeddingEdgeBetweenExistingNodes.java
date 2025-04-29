package lts.pattern;

public class EmbeddingEdgeBetweenExistingNodes extends EmbeddingEdge {

	private final EmbeddingEdge parent;
	
	public EmbeddingEdgeBetweenExistingNodes(EmbeddingEdge parent) {
		super();
		this.parent = parent;
	}

	@Override
	public EmbeddingEdge getParent() {
		return parent;
	}

	@Override
	public EdgeType getType() {
		return EdgeType.EXISTING_NODES;
	}
	
}
