package lts.algorithm.lts;

import java.util.Collection;
import java.util.HashSet;
import java.util.List;

import lts.algorithm.BestPatternsTracker;
import lts.algorithm.ScoreCalculator;
import lts.algorithm.search_history.SearchHistory;
import lts.algorithm.search_history.UpperBoundTreeNode;
import lts.graph.GraphsContext;
import lts.pattern.CCAMCode;
import lts.pattern.Pattern;
import lts.pattern.PatternFactory;


public abstract class LtsRunner {

	protected final GraphsContext graphsContext;
	protected final BestPatternsTracker bestPatterns;
	protected final SearchHistory history;
	
	public LtsRunner(GraphsContext graphsContext, BestPatternsTracker bestPatterns, SearchHistory history) {
		this.graphsContext = graphsContext;
		this.bestPatterns = bestPatterns;
		this.history = history;
	}
	
	/**Adds all children whose CCAMs arn't in exisingCCAMs and with >0 positive support to the list.
	 * Also finds descendant nodes in the upper bound tree, which could be null (Meaning the lineage switches over
	 * to using the edge-score table and rough bounds).*/
	protected void addChildren(
			PatternAndHistoryNode patternAndHistoryNode,
			List<PatternAndHistoryNode> accumulationList,
			HashSet<CCAMCode> existingCCAMs) {
		
		Pattern pattern = patternAndHistoryNode.getPattern();
		UpperBoundTreeNode node = patternAndHistoryNode.getHistoryNode();
		
		Collection<PatternFactory> patternChildrenFactories = pattern.generateChildPatternFactories(graphsContext);

		for (PatternFactory patternChildFactory : patternChildrenFactories) {
			if (patternChildFactory.hasAPositiveEmbedding(graphsContext)) {
				CCAMCode ccam = patternChildFactory.generateCCAMCode(graphsContext);
				if (existingCCAMs.add(ccam)) {
					Pattern patternChild = patternChildFactory.generatePattern();
					double score = ScoreCalculator.calculateScore(patternChild, graphsContext);
					UpperBoundTreeNode nodeChild = null;
					nodeChild = node == null ? null : history.getChildNode(node, score);
					accumulationList.add(new PatternAndHistoryNode(patternChild, nodeChild));
				}
			}
		}
	}

	public abstract void run();
	
}
