package lts.algorithm.search_history;

import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;

import lts.algorithm.ScoreCalculator;


/**Used to construct the history from fast probe*/
public class SearchHistoryFactory {

	private final double binSize;
	private final UpperBoundTreeBuilderNode tree = new UpperBoundTreeBuilderNode();
	
	public SearchHistoryFactory(double binSize) {
		this.binSize = binSize;
	}
	
	public UpperBoundTreeBuilderNode getTreeRoot() {
		return tree;
	}
	
	public UpperBoundTreeBuilderNode addHistory(UpperBoundTreeBuilderNode parentNode, double score) {
		int bin = ScoreCalculator.scoreToBin(score, binSize);
		return parentNode.addChild(bin);
 	}
	
	public SearchHistory generateSearchHistory() {
		HashMap<EdgeAndScoreBin, Integer> edgesAndScoreToBoundMap = new HashMap<>();
		UpperBoundTreeNode upperBoundTree = makeUpperBoundTreeNode(tree, edgesAndScoreToBoundMap, 0, 0);
		SearchHistory history = new SearchHistory(binSize, edgesAndScoreToBoundMap, upperBoundTree);
		return history;
	}
	
	// recursive call to get and record maximum score of descendant nodes
	// generates the history table at the same time
	private UpperBoundTreeNode makeUpperBoundTreeNode(
			UpperBoundTreeBuilderNode node, HashMap<EdgeAndScoreBin, Integer> table, int scoreBin, int edges) {
		int upperBoundBin = scoreBin;
		TreeMap<Integer, UpperBoundTreeBuilderNode> childrenMap = node.getChildren();
		int[] childrenIndex = new int[childrenMap.size()];
		UpperBoundTreeNode[] children = new UpperBoundTreeNode[childrenMap.size()];
		
		int i = 0;
		for (Map.Entry<Integer, UpperBoundTreeBuilderNode> binAndChild : childrenMap.entrySet()) {
			int childBin = binAndChild.getKey();
			UpperBoundTreeNode childNode = makeUpperBoundTreeNode(binAndChild.getValue(), table, childBin, edges + 1);
			if (upperBoundBin < childNode.getUpperBoundBin()) upperBoundBin = childNode.getUpperBoundBin();
			childrenIndex[i] = binAndChild.getKey();
			children[i] = childNode;
			i++;
		}
		
		if (edges > 1) {  // no lookup is ever needed on 0 or 1 edge patterns
			EdgeAndScoreBin tableKey = new EdgeAndScoreBin();
			tableKey.setEdges(edges);
			tableKey.setScoreBin(scoreBin);
			Integer oldUpperBoundForTableEntry = table.get(tableKey);
			if (oldUpperBoundForTableEntry == null || oldUpperBoundForTableEntry < upperBoundBin) {
				table.put(tableKey, upperBoundBin);	
			}
		}
		
		return new UpperBoundTreeNode(upperBoundBin, childrenIndex, children);
	}
	
}
