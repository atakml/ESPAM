package lts.pattern;

public class PatternEdgeBetweenExistingNodes extends PatternEdge {

	private final PatternEdge parent;
											// deeper node is placed second
	private final int firstExistingNodeIndex, secondExistingNodeIndex;

	public PatternEdgeBetweenExistingNodes(PatternEdge parent, int firstExistingNodeIndex, int secondExistingNodeIndex) {
		this.parent = parent;
		this.firstExistingNodeIndex = firstExistingNodeIndex;
		this.secondExistingNodeIndex = secondExistingNodeIndex;
	}

	@Override
	public PatternEdge getParent() {
		return parent;
	}

	@Override
	public EdgeType getType() {
		return EdgeType.EXISTING_NODES;
	}

	public int getFirstExistingNodeIndex() {
		return firstExistingNodeIndex;
	}

	public int getSecondExistingNodeIndex() {
		return secondExistingNodeIndex;
	}

}
