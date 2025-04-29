package lts.algorithm.lts;

import java.util.ArrayList;

/**Represents a layer in the search process of breadth-first LTS.
 * Binds an edge count and a collection of Patterns with their associated UpperBoundTreeNodes.
 * The edge count is used for looking up the upper bound through the history table.
 */
public class LtsEdgesAndExplorationGroup {

	private final int edges;
	private final ArrayList<PatternAndHistoryNode> explorationGroup;
	
	public LtsEdgesAndExplorationGroup(int edges, ArrayList<PatternAndHistoryNode> explorationGroup) {
		this.edges = edges;
		this.explorationGroup = explorationGroup;
	}

	public int getEdges() {
		return edges;
	}

	public ArrayList<PatternAndHistoryNode> getExplorationGroup() {
		return explorationGroup;
	} 
	
}
