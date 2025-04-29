package lts.algorithm.fastprobe;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;

import lts.algorithm.BestPatternsTracker;
import lts.algorithm.ScoreCalculator;
import lts.algorithm.search_history.SearchHistoryFactory;
import lts.algorithm.search_history.UpperBoundTreeBuilderNode;
import lts.graph.GraphsContext;
import lts.pattern.CCAMCode;
import lts.pattern.Pattern;
import lts.pattern.PatternFactory;
import util.logging.Log;


/**Runs Fast Probe*/
public class FastProbeRunner {

	private ArrayList<PatternAndHistoryBuilderNode> group;
	private final GraphsContext graphsContext;
	private final BestPatternsTracker bestPatterns;
	private final SearchHistoryFactory historyFactory;
	private int total = 0;
	private int currEdgesInPatterns = 1;
	
	/**This object should only be run once, else state is undefined.
	 * @param initialGroup Initial 1 edge patterns, bound with UppderBoundTreeBuildeNodes 
	 * which can be null if not generating history.
	 * @param graphsContext
	 * @param bestPatterns The best patterns tracker should be empty.
	 * @param historyFactory Can be null if not generating history.
	 */
	public FastProbeRunner(
			ArrayList<PatternAndHistoryBuilderNode> initialGroup, 
			GraphsContext graphsContext,
			BestPatternsTracker bestPatterns,
			SearchHistoryFactory historyFactory) {
		this.group = initialGroup;
		this.graphsContext = graphsContext;
		this.bestPatterns = bestPatterns;
		this.historyFactory = historyFactory;
	}

	/**Runs fast-probe. This object should only be run once, else state is undefined.
	 * The BestPatternsTracker and SearchHistoryFactory passed into the constructor is updated after this.*/
	public void run() {
		while (group != null) {
			group = explore(group);
		}
		Log.info("FAST PROBE total = "+total);
	}
	
	private ArrayList<PatternAndHistoryBuilderNode> explore(ArrayList<PatternAndHistoryBuilderNode> group) {
		HashSet<CCAMCode> existingCCAMs = new HashSet<>();
		ArrayList<PatternAndHistoryBuilderNode> oldGroup = group;
		ArrayList<PatternAndHistoryBuilderNode> newGroup = new ArrayList<>();
		for (int i = 0; i < oldGroup.size(); i++) {
			PatternAndHistoryBuilderNode patternAndHistoryBuilderNode = oldGroup.get(i);
			oldGroup.set(i, null);

			if(bestPatterns.track(patternAndHistoryBuilderNode.getPattern())) {
				addChildren(patternAndHistoryBuilderNode, newGroup, existingCCAMs);
			}
		}
		Log.info(String.format("FAST PROBE extending %d-edge patterns: %d -> %d",
				currEdgesInPatterns++, oldGroup.size(), newGroup.size()));
		total += oldGroup.size();
		return newGroup.isEmpty() ? null : newGroup;
	}
	
	private void addChildren(
			PatternAndHistoryBuilderNode patternAndHistoryBuilderNode,
			List<PatternAndHistoryBuilderNode> accumulationList,
			HashSet<CCAMCode> existingCCAMs) {
		
		Pattern pattern = patternAndHistoryBuilderNode.getPattern();
		UpperBoundTreeBuilderNode historyBuilderNode = patternAndHistoryBuilderNode.getHistoryBuilderNode();
		Collection<PatternFactory> patternChildrenFactories = pattern.generateChildPatternFactories(graphsContext);
		
		for (PatternFactory patternChildFactory : patternChildrenFactories) {
			if (patternChildFactory.hasAPositiveEmbedding(graphsContext)) {  // if 0 positive support, throw away
				CCAMCode ccam = patternChildFactory.generateCCAMCode(graphsContext);
				if (existingCCAMs.add(ccam)) {
					Pattern patternChild = patternChildFactory.generatePattern();
										
					double score = ScoreCalculator.calculateScore(patternChild, graphsContext);
					UpperBoundTreeBuilderNode nodeChild = null;
					if (historyFactory != null) {
						nodeChild = historyFactory.addHistory(historyBuilderNode, score);
					}
					accumulationList.add(new PatternAndHistoryBuilderNode(patternChild, nodeChild));
				}
			}
		}
	}

}
