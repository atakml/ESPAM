package lts.pattern;

public class PatternExtensionToNewNode implements PatternExtension {

	private final int patternIndex, nodeLabel, edgeLabel;

	public PatternExtensionToNewNode(int patternIndex, int nodeLabel, int edgeLabel) {
		this.patternIndex = patternIndex;
		this.nodeLabel = nodeLabel;
		this.edgeLabel = edgeLabel;
	}

	@Override
	public EdgeType getType() {
		return EdgeType.NEW_NODE;
	}
	
	public int getPatternIndex() {
		return patternIndex;
	}

	public int getNodeLabel() {
		return nodeLabel;
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
		result = prime * result + nodeLabel;
		result = prime * result + patternIndex;
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
		PatternExtensionToNewNode other = (PatternExtensionToNewNode) obj;
		if (edgeLabel != other.edgeLabel)
			return false;
		if (nodeLabel != other.nodeLabel)
			return false;
		if (patternIndex != other.patternIndex)
			return false;
		return true;
	}

}
