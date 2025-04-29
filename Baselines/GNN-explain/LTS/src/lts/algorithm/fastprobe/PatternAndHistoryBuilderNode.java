package lts.algorithm.fastprobe;

import lts.algorithm.search_history.UpperBoundTreeBuilderNode;
import lts.pattern.Pattern;

public class PatternAndHistoryBuilderNode {

	private final Pattern pattern;
	private final UpperBoundTreeBuilderNode historyBuilderNode;
	
	public PatternAndHistoryBuilderNode(Pattern pattern, UpperBoundTreeBuilderNode historyBuilderNode) {
		this.pattern = pattern;
		this.historyBuilderNode = historyBuilderNode;
	}

	public Pattern getPattern() {
		return pattern;
	}

	public UpperBoundTreeBuilderNode getHistoryBuilderNode() {
		return historyBuilderNode;
	}
	
}
