package lts.main.input_preprocessing;

import java.io.IOException;

/**The tokens of a single line in the node file.*/
public class NodeFileLine {

	private static final java.util.regex.Pattern whiteSpaceRegex = java.util.regex.Pattern.compile("\\s+");

	private final String graphLabel, graphId, nodeId, nodeLabel;

	public NodeFileLine(String graphLabel, String graphId, String nodeId, String nodeLabel) {
		this.graphLabel = graphLabel;
		this.graphId = graphId;
		this.nodeId = nodeId;
		this.nodeLabel = nodeLabel;
	}

	/**Initializes a NodeFileLine from a line from the file.
	 * @throws IOException if the line was invalid
	 */
	public static NodeFileLine initFromLine(String line) throws IOException {
		String[] tokens = whiteSpaceRegex.split(line.trim());
		if (tokens.length != 4) {
			throw new IOException("Node file lines are expected to have 4 tokens.");
		}
		return new NodeFileLine(tokens[0], tokens[1], tokens[2], tokens[3]);
	}
	
	public String getGraphLabel() {
		return graphLabel;
	}

	public String getGraphId() {
		return graphId;
	}

	public String getNodeId() {
		return nodeId;
	}

	public String getNodeLabel() {
		return nodeLabel;
	}
	
}
