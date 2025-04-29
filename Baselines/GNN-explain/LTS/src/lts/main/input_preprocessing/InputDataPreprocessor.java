package lts.main.input_preprocessing;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import lts.graph.Edge;
import lts.graph.Graph;
import lts.graph.GraphFactory;
import lts.graph.GraphsContext;
import lts.graph.NeighborNodeIdAndEdgeLabel;
import lts.graph.Node;
import lts.pattern.EmbeddingEdgeBase;
import lts.pattern.Pattern;
import lts.pattern.PatternExtensionBase;
import lts.pattern.PatternExtensionEmbeddingAccumulator;
import lts.pattern.PatternFactory;


/**
 * Provides static functions to pre-process input data for LTS.
 * @author Robert
 *
 */
public class InputDataPreprocessor {

	private static final java.util.regex.Pattern whiteSpaceRegex = java.util.regex.Pattern.compile("\\s+");
	
	public static List<NodeFileLine> readNodeFile(String path) throws IOException {
		List<NodeFileLine> nodeLines = new ArrayList<>();
		try (BufferedReader br = new BufferedReader(new FileReader(path))) {
			String line;
			while ((line = br.readLine()) != null && !line.isEmpty()) {
				String[] tokens = whiteSpaceRegex.split(line.trim());  // split on whitespace
				if (tokens.length != 4) throw new IOException("Node file lines are expected to have 4 tokens.");
				nodeLines.add(new NodeFileLine(tokens[0], tokens[1], tokens[2], tokens[3]));
			}
		}
		return nodeLines;
	}

	public static List<EdgeFileLine> readEdgeFile(String path) throws IOException {
		List<EdgeFileLine> edgeLines = new ArrayList<>();
		try (BufferedReader br = new BufferedReader(new FileReader(path))) {
			String line;
			while ((line = br.readLine()) != null && !line.isEmpty()) {
				String[] tokens = whiteSpaceRegex.split(line.trim()); // split on whitespace
				if (tokens.length != 5) throw new IOException("Edge file lines are expected to have 5 tokens.");
				edgeLines.add(new EdgeFileLine(tokens[0], tokens[1], tokens[2], tokens[3], tokens[4]));
			}
		}
		return edgeLines;
	}
	
	public static List<Node> getNodeListAndAddToEncoder(List<NodeFileLine> nodeFileLines, StringToIntEncoder encoder) {
		List<Node> nodes = new ArrayList<>();
		
		for (NodeFileLine nodeFileLine : nodeFileLines) {
			Node node = new Node();
			node.setGraphId(encoder.encodeGraphId(nodeFileLine.getGraphId()));
			node.setNodeId(encoder.encodeNodeIdForGraph(node.getGraphId(), nodeFileLine.getNodeId()));
			node.setLabel(encoder.encodeNodeLabel(nodeFileLine.getNodeLabel()));
			nodes.add(node);
		}
		
		return nodes;
	}
	
	public static List<Edge> getEdgeListAndAddToEncoder(List<EdgeFileLine> edgeFileLines, StringToIntEncoder encoder) {
		List<Edge> edges = new ArrayList<>();
		
		for (EdgeFileLine edgeFileLine : edgeFileLines) {
			Edge edge = new Edge();
			int graphId = encoder.verifyGraphId(edgeFileLine.getGraphId());
			if (graphId == -1) throw new IllegalArgumentException("Edge declares invalid graph id : "+edgeFileLine.getGraphId());
			edge.setGraphId(graphId);
			for (int i = 0; i < 2; i++) {
				String node = i == 0 ? edgeFileLine.getNode1Id() : edgeFileLine.getNode2Id();
				int nodeId = encoder.verifyNodeIdForGraph(graphId, node);
				if (nodeId == -1) throw new IllegalArgumentException("Edge declares invalid node id for graph "+graphId+" : "+node);
				if (i == 0) edge.setNodeId1(nodeId);
				else edge.setNodeId2(nodeId);
			}
			edge.setLabel(encoder.encodeEdgeLabel(edgeFileLine.getNodeLabel()));
			edges.add(edge);
		}
		
		return edges;
	}
	
	public static GraphsContext generateGraphsContext(List<Node> nodes, List<Edge> edges, int positiveGraphs, int numGraphs) {
		GraphFactory[] graphFactories = new GraphFactory[numGraphs];
		for (int i = 0; i < numGraphs; i++) {
			graphFactories[i] = new GraphFactory();
		}
		
		for (Node node : nodes) {
			graphFactories[node.getGraphId()].addNodeLabel(node.getNodeId(), node.getLabel());
		}
		
		for (Edge edge : edges) {
			graphFactories[edge.getGraphId()].addEdge(edge.getNodeId1(), edge.getNodeId2(), edge.getLabel());
		}
		
		Graph[] graphs = new Graph[numGraphs];
		for (int i = 0; i < numGraphs; i++) {
			graphs[i] = graphFactories[i].generateGraph();
		}
	
		return new GraphsContext(graphs, positiveGraphs);
	}
	
	public static List<Pattern> initializeSingleEdgePatternsWithPositiveEmbeddings(GraphsContext graphsContext) {
		PatternExtensionEmbeddingAccumulator accumulator = new PatternExtensionEmbeddingAccumulator(null);
		for (int graphId = 0; graphId < graphsContext.getNumGraphs(); graphId++) {
			Graph graph = graphsContext.getGraph(graphId);
			for (int nodeId = 0; nodeId < graph.numNodes(); nodeId++) {
				for (NeighborNodeIdAndEdgeLabel edgeInfo : graph.getNeighbors(nodeId)) {
					int nodeLabel = graph.getNodeLabel(nodeId);
					int neighborNodeLabel = graph.getNodeLabel(edgeInfo.getToNodeId());
					if (nodeLabel <= neighborNodeLabel) {
						// if the node labels are equal, both patterns should be generated to handle symmetric subgraphs correctly
						// otherwise we generate only one subgraph
						PatternExtensionBase baseProposal = new PatternExtensionBase(
								nodeLabel, neighborNodeLabel, edgeInfo.getEdgeLabel());
						accumulator.addProposal(baseProposal, graphId, new EmbeddingEdgeBase(nodeId, edgeInfo.getToNodeId()));
					}
				}
			}
		}
		Collection<PatternFactory> patternFactories = accumulator.getPatternFactories();
		return filterOutZeroPositiveSupportPatterns(patternFactories, graphsContext);
	}
	
	// no point in keeping subgraphs with no positive support at all
	private static List<Pattern> filterOutZeroPositiveSupportPatterns(Collection<PatternFactory> patternFactories, GraphsContext graphsContext) {
		List<Pattern> filteredPatterns = new ArrayList<>();
		for (PatternFactory patternFactory : patternFactories) {
			if (patternFactory.hasAPositiveEmbedding(graphsContext)) {
				filteredPatterns.add(patternFactory.generatePattern());
			}
		}
		return filteredPatterns;
	}
}
