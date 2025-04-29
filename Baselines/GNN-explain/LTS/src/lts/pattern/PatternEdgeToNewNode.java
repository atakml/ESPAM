package lts.pattern;

public class PatternEdgeToNewNode extends PatternEdge {

	private final PatternEdge parent;
	private final int existingNodeIndex;
	
	public PatternEdgeToNewNode(PatternEdge parent, int existingNodeIndex) {
		this.parent = parent;
		this.existingNodeIndex = existingNodeIndex;
	}

	@Override
	public PatternEdge getParent() {
		return parent;
	}

	@Override
	public EdgeType getType() {
		return EdgeType.NEW_NODE;
	}

	public int getExistingNodeIndex() {
		return existingNodeIndex;
	}

}
