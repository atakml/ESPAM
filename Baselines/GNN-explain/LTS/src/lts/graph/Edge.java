package lts.graph;

/**Represents an edge read from the file, with all data encoded as ints by StringToIntEncoder.*/
public class Edge {

	private int graphId, nodeId1, nodeId2, label;

	public int getGraphId() {
		return graphId;
	}

	public void setGraphId(int graphId) {
		this.graphId = graphId;
	}

	public int getNodeId1() {
		return nodeId1;
	}

	public void setNodeId1(int nodeId1) {
		this.nodeId1 = nodeId1;
	}

	public int getNodeId2() {
		return nodeId2;
	}

	public void setNodeId2(int nodeId2) {
		this.nodeId2 = nodeId2;
	}

	public int getLabel() {
		return label;
	}

	public void setLabel(int label) {
		this.label = label;
	}
	
}
