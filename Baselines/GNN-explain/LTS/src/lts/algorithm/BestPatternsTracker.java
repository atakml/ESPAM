package lts.algorithm;

import java.util.Arrays;

import lts.graph.GraphsContext;
import lts.pattern.Pattern;
import util.collections.IntArrayView;

/**Tracks the best Patterns and their scores for each positive graph. */
public class BestPatternsTracker {
	private final GraphsContext graphsContext;
	private final Pattern[] bestPatternsForGraphs;
	private final double[] scores;
	
	public BestPatternsTracker(GraphsContext graphsContext) {
		this.graphsContext = graphsContext;
		bestPatternsForGraphs = new Pattern[graphsContext.getNumPositiveGraphs()];
		scores = new double[graphsContext.getNumPositiveGraphs()];
		Arrays.fill(scores, Double.NEGATIVE_INFINITY);
	}
	
	/**Test score of p against current best patterns, replacing them if p is better.
	 * @param p
	 * @return true iff p replaced the best pattern for some positive graph
	 */
	public boolean track(Pattern p) {
		double score = ScoreCalculator.calculateScore(p, graphsContext);
		return track(p, score);
	}
	
	private boolean track(Pattern p, double score) {
		boolean updated = false;
		IntArrayView graphIds = p.getEmbeddingGraphIds();
		for (int i = 0; i < graphIds.length(); i++) {
			int graphId = graphIds.get(i);
			if (graphId >= graphsContext.getNumPositiveGraphs()) break;
			
			if (scores[graphId] < score) {
				scores[graphId] = score;
				bestPatternsForGraphs[graphId] = p;
				updated = true;
			}
		}
		return updated;
	}
	
	
	/**Test score of p against current best patterns, replacing them if p is better.
	 * At the same time test if upperBound is promising.
	 * @param p
	 * @param upperBound
	 * @return true iff the upperBound is promising
	 */
	public boolean trackAndcheckPromising(Pattern p, double upperBound) {
		double score = ScoreCalculator.calculateScore(p, graphsContext);
		return trackAndcheckPromising(p, score, upperBound);
	}
	
	private boolean trackAndcheckPromising(Pattern p, double score, double upperBound) {
		boolean promising = false;
		IntArrayView graphIds = p.getEmbeddingGraphIds();
		for (int i = 0; i < graphIds.length(); i++) {
			int graphId = graphIds.get(i);
			if (graphId >= graphsContext.getNumPositiveGraphs()) break;
			
			if (scores[graphId] < upperBound) {
				promising = true;
			}
			
			if (scores[graphId] < score) {
				scores[graphId] = score;
				bestPatternsForGraphs[graphId] = p;
			}
		}
		return promising;
	}
	
	public Pattern getBestPatternForGraph(int graphId) {
		return bestPatternsForGraphs[graphId];
	}
	
	public double getBestScoreForGraph(int graphId) {
		return scores[graphId];
	}
	
	public int size() {
		return bestPatternsForGraphs.length;
	}
}
