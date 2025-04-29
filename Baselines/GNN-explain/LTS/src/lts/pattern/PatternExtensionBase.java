package lts.pattern;

public class PatternExtensionBase implements PatternExtension {

	private final int node1Label, node2Label, edgeLabel;
	
	public PatternExtensionBase(int node1Label, int node2Label, int edgeLabel) {
		this.node1Label = node1Label;
		this.node2Label = node2Label;
		this.edgeLabel = edgeLabel;
	}

	@Override
	public EdgeType getType() {
		return EdgeType.BASE;
	}

	public int getNode1Label() {
		return node1Label;
	}

	public int getNode2Label() {
		return node2Label;
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
		result = prime * result + node1Label;
		result = prime * result + node2Label;
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
		PatternExtensionBase other = (PatternExtensionBase) obj;
		if (edgeLabel != other.edgeLabel)
			return false;
		if (node1Label != other.node1Label)
			return false;
		if (node2Label != other.node2Label)
			return false;
		return true;
	}

}
