package lts.algorithm.lts;

import lts.algorithm.search_history.UpperBoundTreeNode;
import lts.pattern.Pattern;

public class PatternAndHistoryNode {

	private final Pattern pattern;
	private final UpperBoundTreeNode historyNode;
	
	public PatternAndHistoryNode(Pattern pattern, UpperBoundTreeNode historyNode) {
		this.pattern = pattern;
		this.historyNode = historyNode;
	}

	public Pattern getPattern() {
		return pattern;
	}

	public UpperBoundTreeNode getHistoryNode() {
		return historyNode;
	}
	
}
