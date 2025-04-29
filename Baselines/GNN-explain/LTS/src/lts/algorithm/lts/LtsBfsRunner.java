package lts.algorithm.lts;

import java.util.ArrayList;
import java.util.HashSet;

import lts.algorithm.BestPatternsTracker;
import lts.algorithm.search_history.SearchHistory;
import lts.algorithm.search_history.UpperBoundTreeNode;
import lts.graph.GraphsContext;
import lts.pattern.CCAMCode;
import lts.pattern.Pattern;
import util.logging.Log;

/**Runs LTS breadth first.*/
public class LtsBfsRunner extends LtsRunner {

	private LtsEdgesAndExplorationGroup group;
	private int total = 0;
	
	/**This object should only be run once, else state is undefined.
	 * @param initalGroup initial 1 edge patterns, starting at 1 edge
	 * @param graphsContext
	 * @param bestPatterns this should have already been run through fast probe
	 * @param history generated from a SearchHistoryFactory populated by fast probe
	 */
	public LtsBfsRunner(
			ArrayList<PatternAndHistoryNode> initialGroup,
			GraphsContext graphsContext,
			BestPatternsTracker bestPatterns,
			SearchHistory history) {
		super(graphsContext, bestPatterns, history);
		this.group = new LtsEdgesAndExplorationGroup(1, initialGroup);  // initial patterns have 1 edge
	}
	
	/**This object should only be run once, else state is undefined. */
	@Override
	public void run() {
		while (group != null) {
			group = explore(group);
		}
		Log.info("LTS-BFS total = "+total);
	}
	
	private LtsEdgesAndExplorationGroup explore(LtsEdgesAndExplorationGroup group) {
		HashSet<CCAMCode> existingCCAMs = new HashSet<>();
		ArrayList<PatternAndHistoryNode> oldGroup = group.getExplorationGroup();
		ArrayList<PatternAndHistoryNode> newGroup = new ArrayList<>();
		for (int i = 0; i < oldGroup.size(); i++) {
			PatternAndHistoryNode patternAndHistoryNode = oldGroup.get(i);
			Pattern pattern = patternAndHistoryNode.getPattern();
			UpperBoundTreeNode node = patternAndHistoryNode.getHistoryNode();
			oldGroup.set(i, null);  // frees up memory
			
			double upperBound = history.predictUpperBound(node, graphsContext, pattern, group.getEdges());
			
			boolean isPromising = bestPatterns.trackAndcheckPromising(pattern, upperBound);
			if (isPromising) {
				addChildren(patternAndHistoryNode, newGroup, existingCCAMs);
			}
		}
		Log.info(String.format("LTS-BFS extending %d-edge patterns: %d -> %d",
				group.getEdges(), oldGroup.size(), newGroup.size()));
		total += oldGroup.size();
		return newGroup.isEmpty() ? null : new LtsEdgesAndExplorationGroup( group.getEdges() + 1, newGroup);
	}
	
}