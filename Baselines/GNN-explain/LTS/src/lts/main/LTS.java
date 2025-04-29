package lts.main;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Random;

import lts.algorithm.BestPatternsTracker;
import lts.algorithm.ScoreCalculator;
import lts.algorithm.fastprobe.FastProbeRunner;
import lts.algorithm.fastprobe.PatternAndHistoryBuilderNode;
import lts.algorithm.lts.LtsBfsRunner;
import lts.algorithm.lts.LtsDfsRunner;
import lts.algorithm.lts.LtsRunner;
import lts.algorithm.lts.PatternAndHistoryNode;
import lts.algorithm.search_history.SearchHistory;
import lts.algorithm.search_history.SearchHistoryFactory;
import lts.graph.Edge;
import lts.graph.GraphsContext;
import lts.graph.Node;
import lts.main.input_preprocessing.AlgorithmType;
import lts.main.input_preprocessing.EdgeFileLine;
import lts.main.input_preprocessing.InputDataPreprocessor;
import lts.main.input_preprocessing.IntToStringDecoder;
import lts.main.input_preprocessing.NodeFileLine;
import lts.main.input_preprocessing.ProgramParameters;
import lts.main.input_preprocessing.StringToIntEncoder;
import lts.pattern.Pattern;
import util.logging.Log;
import util.string.StringUtil;
import util.time.StopWatch;

// Program arguments handled in main.input_preprocessing.ProgramParameters.java
public class LTS {

	// default score bin interval width for fast probe's history
	private static final double DEFAULT_BIN_SIZE = 0.1;
	
	public static void main(String[] args) {

		ProgramParameters params = ProgramParameters.initFromArgsIfErrPrint(args);
		if (params == null) System.exit(-1);
		
		List<NodeFileLine> nodeFileLines;
		List<EdgeFileLine> edgeFileLines;
		Log.info("Reading input files.");
		try {
			nodeFileLines = InputDataPreprocessor.readNodeFile(params.getNodeFileName());
			edgeFileLines = InputDataPreprocessor.readEdgeFile(params.getEdgeFileName());
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
		
		Log.info("Constructing graphs.");
		StringToIntEncoder encoder = new StringToIntEncoder();
		List<Node> nodes = InputDataPreprocessor.getNodeListAndAddToEncoder(nodeFileLines, encoder);
		List<Edge> edges = InputDataPreprocessor.getEdgeListAndAddToEncoder(edgeFileLines, encoder);
		IntToStringDecoder decoder = encoder.generateDecoder();
	
		GraphsContext graphsContext = InputDataPreprocessor.generateGraphsContext(nodes, edges, params.getPositiveGraphs(), encoder.numGraphIds());
	
		Log.info("Constructing initial patterns.");
		List<Pattern> initialPatterns = 
				InputDataPreprocessor.initializeSingleEdgePatternsWithPositiveEmbeddings(graphsContext);
		
		if (params.getSeed() != null) {
			Collections.shuffle(initialPatterns, new Random(params.getSeed()));
		}
		
		String outFileName;
		if (params.getOutputFileName() == null) {
			outFileName = new File(params.getNodeFileName()).getName();
		} else {
			outFileName = params.getOutputFileName();
		}

		if (params.isDateTimeInfo()) {
			outFileName += " "+new Date().toString().replace(':', '_');
		}

		Log.info("Starting stopwatch.");
		StopWatch stopWatch = new StopWatch();
		
		// FAST PROBE
		Log.info("Running FAST PROBE");
		
		double binSize = params.getBinSize() == null ? DEFAULT_BIN_SIZE : params.getBinSize();
		
		SearchHistoryFactory historyFactory = null;
		if (params.getAlgorithm() == AlgorithmType.LTSBFS || params.getAlgorithm() == AlgorithmType.LTSDFS) {
			// if is lts, we're going to need fastprobe's history
			historyFactory = new SearchHistoryFactory(binSize);
		}
		
		// Pair initial patterns with tree builder nodes to build history tree to guide lts
		ArrayList<PatternAndHistoryBuilderNode> initialPatternsWithTreeBuilderNodes = new ArrayList<>(initialPatterns.size());
		for (Pattern initialPattern : initialPatterns) {
			initialPatternsWithTreeBuilderNodes.add(new PatternAndHistoryBuilderNode(
					initialPattern,
					historyFactory == null ? null : historyFactory.addHistory(  // if no history factory then don't need tree builder nodes
							historyFactory.getTreeRoot(),
							ScoreCalculator.calculateScore(initialPattern, graphsContext))));
		}
		
		BestPatternsTracker bestPatterns = new BestPatternsTracker(graphsContext);
		FastProbeRunner fpRunner = new FastProbeRunner(
				initialPatternsWithTreeBuilderNodes,
				graphsContext, bestPatterns, historyFactory);

		fpRunner.run();
		
		
		Log.info("Writing FAST PROBE output to file.");
		try {
			printToOutput(args, "FAST PROBE "+outFileName, bestPatterns, graphsContext, decoder);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		if (params.getAlgorithm() == AlgorithmType.LTSBFS || params.getAlgorithm() == AlgorithmType.LTSDFS) {
			Log.info("Running "+params.getAlgorithm());

			SearchHistory history = historyFactory.generateSearchHistory();
			
			// System.out.println(history.toString());  // for debug
			ArrayList<PatternAndHistoryNode> initialPatternsWithTreeNodes = new ArrayList<>(initialPatterns.size());
			for (Pattern initialPattern : initialPatterns) {
				initialPatternsWithTreeNodes.add(new PatternAndHistoryNode(
						initialPattern, 
						history.getChildNode(
								history.getRootNode(),
								ScoreCalculator.calculateScore(initialPattern, graphsContext))));
			}

			LtsRunner ltsRunner;
			if (params.getAlgorithm() == AlgorithmType.LTSBFS) {
				ltsRunner = new LtsBfsRunner(
						initialPatternsWithTreeNodes,
						graphsContext, bestPatterns, history); 
			} else {
				ltsRunner = new LtsDfsRunner(
						initialPatternsWithTreeNodes,
						graphsContext, bestPatterns, history); 
			}

			ltsRunner.run();
			
			Log.info("Writing "+params.getAlgorithm()+" output to file.");
			try {
				printToOutput(args, params.getAlgorithm()+" "+outFileName, bestPatterns, graphsContext, decoder);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		Log.info("Pausing stopwatch.");
		stopWatch.pause();
		Log.info("Time elapsed = "+stopWatch.getTimeMillis()+"ms");
	}
	
	/**Prints the best patterns to output file*/
	private static void printToOutput(
			String[] args,
			String out,
			BestPatternsTracker bestPatterns,
			GraphsContext graphsContext,
			IntToStringDecoder decoder) throws IOException {
		File resultsFolder = new File(".\\results");
		if (!resultsFolder.isDirectory() && !resultsFolder.mkdirs()) {
			throw new IOException("Could not create .\\results folder.");
		}
		try (BufferedWriter bw = new BufferedWriter(new FileWriter(".\\results\\"+out))) {
			bw.write(String.format("Program args = %s%n", StringUtil.toSeparatedString(" ", args)));
			double scoreTotal = 0.0;
			for (int graphId = 0; graphId < graphsContext.getNumPositiveGraphs(); graphId++) {
				bw.write(String.format("== BEST PATTERN FOR GRAPH %s ==%n", decoder.decodeGraphId(graphId)));
				bw.write(bestPatterns.getBestPatternForGraph(graphId).toString(graphsContext, decoder));
				bw.write('\n');
				double score = ScoreCalculator.calculateScore(bestPatterns.getBestPatternForGraph(graphId), graphsContext);
				scoreTotal += score;
				bw.write(String.format("Score: %f%n", score));
				bw.write("\n");
			}
			bw.write("== SUMMARY ==\n");
			bw.write(String.format("Average Score: %f%n", scoreTotal / graphsContext.getNumPositiveGraphs()));
		}
		Log.info("Wrote output file: "+out);
	}
	
}
