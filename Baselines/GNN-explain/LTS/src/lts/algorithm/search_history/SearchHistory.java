package lts.algorithm.search_history;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import lts.algorithm.ScoreCalculator;
import lts.graph.GraphsContext;
import lts.pattern.Pattern;


/**The history of a fast probe run, used to predict upper bounds for scores in LTS.*/
public class SearchHistory {

	private final double binSize;
	private final HashMap<EdgeAndScoreBin, Integer> table;
	private final UpperBoundTreeNode tree;
	
	protected SearchHistory(double binSize,	HashMap<EdgeAndScoreBin, Integer> table, UpperBoundTreeNode tree) {
		this.binSize = binSize;
		this.table = table;
		this.tree = tree;
	}
	
	public UpperBoundTreeNode getRootNode() {
		return tree;
	}
	
	/**Returns the child of the node for the score. Returns null if no such child exists.*/
	public UpperBoundTreeNode getChildNode(UpperBoundTreeNode parent, double score) {
		int scoreBin = ScoreCalculator.scoreToBin(score, binSize);
		return parent.getChild(scoreBin);
	}
	
	public double getUpperBound(UpperBoundTreeNode node) {
		return ScoreCalculator.binToScore(node.getUpperBoundBin(), binSize);
	}
	
	/**Uses the number of edges and score to predict from the history table.
	 * Returns NaN if entry not found in the table.*/
	public double predictBoundFromTable(int edges, double score) {
		EdgeAndScoreBin key = new EdgeAndScoreBin();
		key.setEdges(edges);
		key.setScoreBin(ScoreCalculator.scoreToBin(score, binSize));
		Integer bin = table.get(key);
		return bin == null ? Double.NaN : ScoreCalculator.binToScore(bin, binSize);
	}
	
	private static final Comparator<Map.Entry<EdgeAndScoreBin, Integer>> sortTableEntriesComparator =
			new Comparator<Map.Entry<EdgeAndScoreBin, Integer>>() {
		@Override
		public int compare(Entry<EdgeAndScoreBin, Integer> o1, Entry<EdgeAndScoreBin, Integer> o2) {
			return o1.getKey().compareTo(o2.getKey());
		}	
	};
	
	/**Tries to predict the upper bound from the history tree, then table, then from assuming 0 negative embeddings.*/
	public double predictUpperBound(UpperBoundTreeNode node, GraphsContext graphsContext, Pattern pattern, int edges) {
		double upperBound;
		if (node != null) {
			upperBound = getUpperBound(node);
		} else {
			double bound = predictBoundFromTable(
					edges,
					ScoreCalculator.calculateScore(pattern, graphsContext));
			if (!Double.isNaN(bound)) {
				upperBound = bound;
			} else {
				upperBound = ScoreCalculator.calculateScore(
						pattern.getNumPositiveEmbeddings(graphsContext),
						graphsContext.getNumPositiveGraphs(), 
						0,
						graphsContext.getNumGraphs() - graphsContext.getNumPositiveGraphs());
			}
		}
		return upperBound;
	}
	
	public String toString() {
		StringBuilder sb = new StringBuilder();
		ArrayList<Map.Entry<EdgeAndScoreBin, Integer>> entries = new ArrayList<>(table.size());
		for (Map.Entry<EdgeAndScoreBin, Integer> entry : table.entrySet()) {
			entries.add(entry);
		}
		Collections.sort(entries, sortTableEntriesComparator);
		for (Map.Entry<EdgeAndScoreBin, Integer> entry : entries) {
			sb.append(entry.getKey()).append(' ').append(entry.getValue()).append('\n');
		}
		sb.append(tree.toString());
		return sb.toString();
	}
	
}
