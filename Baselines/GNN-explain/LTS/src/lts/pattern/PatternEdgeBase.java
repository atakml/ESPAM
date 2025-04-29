package lts.pattern;

public class PatternEdgeBase extends PatternEdge {

	@Override
	public PatternEdge getParent() {
		return null;
	}

	@Override
	public EdgeType getType() {
		return EdgeType.BASE;
	}

}
