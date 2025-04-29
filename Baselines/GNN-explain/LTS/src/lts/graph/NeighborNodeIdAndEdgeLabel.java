package lts.graph;

/**Used by Graph to represent the neighbor of a node and the edge between them.
 * Does not hold the neighbor node's label because that can be looked up fast through the Graph,
 * once you have the id.*/
public class NeighborNodeIdAndEdgeLabel implements Comparable<NeighborNodeIdAndEdgeLabel> {
	private final int toNodeId, edgeLabel;
	
	public NeighborNodeIdAndEdgeLabel(int toNodeId, int edgeLabel) {
		this.toNodeId = toNodeId;
		this.edgeLabel = edgeLabel;
	}
	
	@Override
	public int compareTo(NeighborNodeIdAndEdgeLabel o) {
		return this.toNodeId - o.toNodeId;
	}

	public int getToNodeId() {
		return toNodeId;
	}

	public int getEdgeLabel() {
		return edgeLabel;
	}
	
}
