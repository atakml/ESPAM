package lts.graph;

/**Hold overall information about the graphs used. */
public class GraphsContext {

	final private Graph[] graphs;
	private final int numPositive;
	
	public GraphsContext(Graph[] graphs, int numPositive) {
		this.graphs = graphs;
		this.numPositive = numPositive;
	}
	
	public int getNumGraphs() {
		return graphs.length;
	}
	
	public Graph getGraph(int i) {
		return graphs[i];
	}
	
	public int getNumPositiveGraphs() {
		return numPositive;
	}
	
	public int getNumNegativeGraphs() {
		return graphs.length - numPositive;
	}
	
}
