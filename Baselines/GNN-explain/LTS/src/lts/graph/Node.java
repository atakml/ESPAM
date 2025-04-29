package lts.graph;

/**Represents a node read from the file, with all data encoded as ints by StringToIntEncoder.*/
public class Node {

	private int graphId, nodeId, label;

	public int getGraphId() {
		return graphId;
	}

	public void setGraphId(int graphId) {
		this.graphId = graphId;
	}

	public int getNodeId() {
		return nodeId;
	}

	public void setNodeId(int nodeId) {
		this.nodeId = nodeId;
	}

	public int getLabel() {
		return label;
	}

	public void setLabel(int label) {
		this.label = label;
	}
	
}
