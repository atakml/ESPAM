package lts.algorithm;

import lts.graph.GraphsContext;
import lts.pattern.Pattern;


public class ScoreCalculator {

	public static double calculateScore(Pattern p, GraphsContext graphsContext) {
		int positive = p.getNumPositiveEmbeddings(graphsContext);
		int positiveTotal = graphsContext.getNumPositiveGraphs();
		int negative = p.getNumEmbeddings() - positive;
		int negativeTotal = graphsContext.getNumGraphs() - positiveTotal;
		return calculateScore(positive, positiveTotal, negative, negativeTotal);
	}
	
	public static double calculateScore(int positive, int positiveTotal, int negative, int negativeTotal) {
		double positiveRatio = ((double) positive) / positiveTotal;
		double negativeRatio = ((double) (negative + 1)) / (negativeTotal + 1);
		return Math.log(positiveRatio / negativeRatio);
	}
	
	public static int scoreToBin(double score, double binSize) {
		return (int) Math.ceil(score / binSize);
	}
	
	public static double binToScore(int bin, double binSize) {
		return bin * binSize;
	}
	
}
