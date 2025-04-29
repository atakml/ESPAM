package lts.graph;

import java.util.Collections;
import java.util.List;

import lts.main.input_preprocessing.IntToStringDecoder;
import util.collections.ArrayView;
import util.collections.CollectionUtil;
import util.collections.CompactDenseListList;
import util.collections.CompactListList;
import util.string.StringUtil;

/**Represents a graph, encoding in ints. Supports label and neighbor lookups. 
 * Holds NO knowledge about whether its positive or not or any other context information.*/
public class Graph {

	private final int[] nodeLabels;
	private final CompactListList<NeighborNodeIdAndEdgeLabel> edgeInfo;

	protected <L extends List<NeighborNodeIdAndEdgeLabel>> Graph(int[] nodeLabels, List<L> neighborNodeIdsAndEdgeLabels) {
		this.nodeLabels = nodeLabels;
		CollectionUtil.extendListToIndex(neighborNodeIdsAndEdgeLabels, nodeLabels.length - 1);

		for (List<NeighborNodeIdAndEdgeLabel> edgeInfosForNode : neighborNodeIdsAndEdgeLabels) {
			if (edgeInfosForNode != null) Collections.sort(edgeInfosForNode);
		}
		
		edgeInfo = new CompactDenseListList<NeighborNodeIdAndEdgeLabel>(neighborNodeIdsAndEdgeLabels);
	}
	
	public int numNodes() {
		return nodeLabels.length;
	}
	
	public int getNodeLabel(int nodeId) {
		return nodeLabels[nodeId];
	}
	
	public ArrayView<NeighborNodeIdAndEdgeLabel> getNeighbors(int nodeId) {
		return edgeInfo.getList(nodeId);
	}
	
	public int getEdgeLabel(int node1Id, int node2Id) {
		int listIndex, targetNode;
		if (edgeInfo.listSize(node1Id) < edgeInfo.listSize(node2Id)) {
			listIndex = node1Id;
			targetNode = node2Id;
		} else {
			listIndex = node2Id;
			targetNode = node1Id;
		}
		ArrayView<NeighborNodeIdAndEdgeLabel> list = edgeInfo.getList(listIndex);
		int index = binarySearch(list, targetNode);
		return index == -1 ? -1 : list.get(index).getEdgeLabel();
	}
	
	// neighbors of a node are sorted by ids, allowing for O(log N) lookup
	private int binarySearch(ArrayView<NeighborNodeIdAndEdgeLabel> array, int target) {
		 int start = 0;
		 int end = array.length();
		 while (start < end) {
			 int mid = start + (end - start) / 2;
			 int currMidNodeId = array.get(mid).getToNodeId();
			 if (target < currMidNodeId) {
				 end = mid;
			 } else if (currMidNodeId < target) {
				 start = mid + 1;
			 } else {
				 return mid;
			 }
		 }
		 return start;
	}
	
	// for debugging
	public String toString(int graphId, IntToStringDecoder decoder) {
		StringBuilder sb = new StringBuilder("== GRAPH ").append(decoder.decodeGraphId(graphId)).append(" ==\n");
		sb.append("Node Labels: \n");
		for (int i = 0; i < nodeLabels.length; i++) {
			sb.append(decoder.decodeNodeIdForGraphs(graphId, i)).append(" : ").append(decoder.decodeNodeLabel(nodeLabels[i])).append('\n');
		}
		sb.append("Edge Labels:\n");
		for (int i = 0; i < edgeInfo.numLists(); i++) {
			sb.append(decoder.decodeNodeIdForGraphs(graphId, i)).append(" : ");
			ArrayView<NeighborNodeIdAndEdgeLabel> edgeInfosForNode = edgeInfo.getList(i);
			String[] strs = new String[edgeInfosForNode.length()];
			for (int j = 0; j < strs.length; j++) {
				strs[j] = new StringBuilder().append(decoder.decodeNodeIdForGraphs(graphId, edgeInfosForNode.get(j).getToNodeId()))
				.append('(').append(decoder.decodeEdgeLabel(edgeInfosForNode.get(j).getEdgeLabel())).append(')').toString();
			}
			sb.append(StringUtil.toSeparatedString(", ", strs)).append('\n');
		}
		return sb.toString();
	}
}
