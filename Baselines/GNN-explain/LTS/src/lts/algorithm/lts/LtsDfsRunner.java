package lts.algorithm.lts;

import java.util.ArrayList;
import java.util.HashSet;

import lts.algorithm.BestPatternsTracker;
import lts.algorithm.search_history.SearchHistory;
import lts.algorithm.search_history.UpperBoundTreeNode;
import lts.graph.GraphsContext;
import lts.pattern.CCAMCode;
import lts.pattern.Pattern;
import util.collections.CollectionUtil;
import util.logging.Log;

/**Runs LTS depth first.*/
public class LtsDfsRunner extends LtsRunner {
	
	private final ArrayList<PatternAndHistoryNode> group;
	private final HashSet<CCAMCode> existingCCAMs = new HashSet<>();
	private final ArrayList<Integer> numPatternsExploredAtEachDepthLayer = new ArrayList<Integer>();
	
	/**This object should only be run once, else state is undefined.
	 * @param initalGroup initial 1 edge patterns
	 * @param graphsContext
	 * @param bestPatterns this should have already been run through fast probe
	 * @param history generated from a SearchHistoryFactory populated by fast probe
	 */
	public LtsDfsRunner(
			ArrayList<PatternAndHistoryNode> initalGroup,
			GraphsContext graphsContext,
			BestPatternsTracker bestPatterns,
			SearchHistory history) {
		super(graphsContext, bestPatterns, history);
		this.group = initalGroup;
	}
	
	/**This object should only be run once, else state is undefined. */
	@Override
	public void run() {
		search(group, 1);
		int total = 0;
		for (int i = 0; i < numPatternsExploredAtEachDepthLayer.size(); i++) {
			int edges = i + 1;
			int numPatternsExplored = numPatternsExploredAtEachDepthLayer.get(i);
			Log.info(String.format("LTS-DFS total # of %d-edge patterns explored = %d",
					edges, numPatternsExplored));
			total += numPatternsExplored;
		}
		Log.info("LTS-DFS total = "+total);
	}

	private void search(ArrayList<PatternAndHistoryNode> oldGroup, int edges) {
		if (oldGroup.isEmpty()) return;
		CollectionUtil.extendListToIndex(numPatternsExploredAtEachDepthLayer, edges - 1);
		Integer prevCount = numPatternsExploredAtEachDepthLayer.get(edges - 1);
		if (prevCount == null) prevCount = 0;
		numPatternsExploredAtEachDepthLayer.set(edges - 1, prevCount + oldGroup.size());
		for (int i = 0; i < oldGroup.size(); i++) {
			PatternAndHistoryNode patternAndHistoryNode = oldGroup.get(i);
			Pattern pattern = patternAndHistoryNode.getPattern();
			UpperBoundTreeNode node = patternAndHistoryNode.getHistoryNode();
			oldGroup.set(i, null);  // frees up memory

			double upperBound = history.predictUpperBound(node, graphsContext, pattern, edges);
			
			ArrayList<PatternAndHistoryNode> newGroup = new ArrayList<>();
			boolean isPromising = bestPatterns.trackAndcheckPromising(pattern, upperBound);
			if (isPromising) {
				addChildren(patternAndHistoryNode, newGroup, existingCCAMs);
			}
			search(newGroup, edges + 1);
		}
	}
	
}
