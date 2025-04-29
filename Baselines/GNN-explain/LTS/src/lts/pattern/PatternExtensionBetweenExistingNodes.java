package lts.pattern;

public class PatternExtensionBetweenExistingNodes implements PatternExtension {

	private final int patternIndex1, patternIndex2, edgeLabel;
	
	public PatternExtensionBetweenExistingNodes(int patternIndex1, int patternIndex2, int edgeLabel) {
		this.patternIndex1 = patternIndex1;
		this.patternIndex2 = patternIndex2;
		this.edgeLabel = edgeLabel;
	}

	@Override
	public EdgeType getType() {
		return EdgeType.EXISTING_NODES;
	}

	public int getPatternIndex1() {
		return patternIndex1;
	}

	public int getPatternIndex2() {
		return patternIndex2;
	}

	@Override
	public int getEdgeLabel() {
		return edgeLabel;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + edgeLabel;
		result = prime * result + patternIndex1;
		result = prime * result + patternIndex2;
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
		PatternExtensionBetweenExistingNodes other = (PatternExtensionBetweenExistingNodes) obj;
		if (edgeLabel != other.edgeLabel)
			return false;
		if (patternIndex1 != other.patternIndex1)
			return false;
		if (patternIndex2 != other.patternIndex2)
			return false;
		return true;
	}

}
