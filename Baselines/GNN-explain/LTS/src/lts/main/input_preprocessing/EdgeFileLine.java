package lts.main.input_preprocessing;

import java.io.IOException;

/**The tokens of a single line in the edge file.*/
public class EdgeFileLine {

	private static final java.util.regex.Pattern whiteSpaceRegex = java.util.regex.Pattern.compile("\\s+");
	
	private final String graphLabel, graphId, node1Id, node2Id, nodeLabel;

	public EdgeFileLine(String graphLabel, String graphId, String node1Id, String node2Id, String nodeLabel) {
		this.graphLabel = graphLabel;
		this.graphId = graphId;
		this.node1Id = node1Id;
		this.node2Id = node2Id;
		this.nodeLabel = nodeLabel;
	}
	
	/**Initializes a EdgeFileLine from a line from the file.
	 * @throws IOException if the line was invalid
	 */
	public static EdgeFileLine initFromLine(String line) throws IOException {
		String[] tokens = whiteSpaceRegex.split(line.trim());
		if (tokens.length != 4) {
			throw new IOException("Edge file lines are expected to have 5 tokens.");
		}
		return new EdgeFileLine(tokens[0], tokens[1], tokens[2], tokens[3], tokens[4]);
	}

	public String getGraphLabel() {
		return graphLabel;
	}

	public String getGraphId() {
		return graphId;
	}

	public String getNode1Id() {
		return node1Id;
	}

	public String getNode2Id() {
		return node2Id;
	}

	public String getNodeLabel() {
		return nodeLabel;
	}
	
}
