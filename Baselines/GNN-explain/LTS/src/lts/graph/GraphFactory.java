package lts.graph;

import java.util.ArrayList;

import util.collections.CollectionUtil;

/**Factory class to constructing graphs.*/
public class GraphFactory {

	private final ArrayList<Integer> nodeLabels = new ArrayList<>();
	private final ArrayList<ArrayList<NeighborNodeIdAndEdgeLabel>> edgeInfos = new ArrayList<>();
	
	public void addNodeLabel(int nodeId, int label) {
		CollectionUtil.extendListToIndex(nodeLabels, nodeId);
		nodeLabels.set(nodeId, label);
	}
	
	/**Order of nodes does not matter, both nodes will remember the other as its neighbor.*/
	public void addEdge(int node1Id, int node2Id, int label) {
		for (int i = 0; i < 2; i++) {
			int fromNode = i == 0 ? node1Id : node2Id;
			int toNode = i == 0 ? node2Id : node1Id;
			CollectionUtil.addItemToListOfArrayLists(edgeInfos, fromNode, new NeighborNodeIdAndEdgeLabel(toNode, label));
		}
	}
	
	public Graph generateGraph() {
		return new Graph(CollectionUtil.toIntArray(nodeLabels), edgeInfos);
	}
	
}
