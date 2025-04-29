package lts.main.input_preprocessing;

import java.util.ArrayList;
import java.util.List;

import util.collections.CollectionUtil;
import util.collections.UniqueIntEncoder;

/**Encodes the ids and labels of the input to ints, for faster logic and more compact internal representations.
 * This class can produce a decoder that does the reverse operation of decoding ints back into Strings. */
public class StringToIntEncoder {

	private final UniqueIntEncoder<String>
	graphIdEncoder = new UniqueIntEncoder<>(),
	nodeLabelEncoder = new UniqueIntEncoder<>(),
	edgeLabelEncoder = new UniqueIntEncoder<>();
	
	private final ArrayList<UniqueIntEncoder<String>> graphIndexedNodeIdEncoders = new ArrayList<>();
	
	/**Encodes a graph id.
	 * @param graphId Either a previously seen graph id or a new graph id. 
	 * @return An int unique to the graph id.
	 */
	public int encodeGraphId(String graphId) {
		return graphIdEncoder.encode(graphId);
	}
	
	/**Retrieves the int for a previously encoded graph id.
	 * @param graphId Either a previously seen graph id or a new graph id. 
	 * @return The int encoding for a previously encoded graph id or -1 for a new graph id.
	 */
	public int verifyGraphId(String graphId) {
		return graphIdEncoder.verify(graphId);
	}
	
	/** @return The number of encoded graph ids.
	 */
	public int numGraphIds() {
		return graphIdEncoder.size();
	}
	
	/**Encodes a node id for a graph.
	 * @param graphId Either a previously seen node id or a new node id for the given graph. 
	 * @return An int unique to the node id within the graph.
	 */
	public int encodeNodeIdForGraph(int graphId, String node) {
		CollectionUtil.extendListToIndex(graphIndexedNodeIdEncoders, graphId);
		UniqueIntEncoder<String> nodeEncoderForGraph = graphIndexedNodeIdEncoders.get(graphId);
		if (nodeEncoderForGraph == null) {
			graphIndexedNodeIdEncoders.set(graphId, nodeEncoderForGraph = new UniqueIntEncoder<String>());
		}
		return nodeEncoderForGraph.encode(node);
	}
	
	/**Retrieves the int for a previously encoded node id for a graph.
	 * @param graphId The id of the graph the node is in.
	 * @param node Either a previously seen node id or a new node id for the given graph. 
	 * @return The int encoding for a previously encoded node id or -1 for a new node id for the given graph.
	 */
	public int verifyNodeIdForGraph(int graphId, String node) {
		if (graphIndexedNodeIdEncoders.size() <= graphId) {
			return -1;
		}
		UniqueIntEncoder<String> nodeEncoderForGraph = graphIndexedNodeIdEncoders.get(graphId);
		if (nodeEncoderForGraph == null) {
			return -1;
		}
		return nodeEncoderForGraph.verify(node);
	}
	
	/**Encodes a node label.
	 * @param graphId Either a previously seen node label or a new node label. 
	 * @return An int unique to the node label.
	 */
	public int encodeNodeLabel(String nodeLabel) {
		return nodeLabelEncoder.encode(nodeLabel);
	}
	
	/**Encodes a edge label.
	 * @param graphId Either a previously seen edge label or a new edge label. 
	 * @return An int unique to the edge label.
	 */
	public int encodeEdgeLabel(String edgeLabel) {
		return edgeLabelEncoder.encode(edgeLabel);
	}

	/**@return A decoder than reverses the operation of this encoder. 
	 * The decoder is only valid for the state of the encoder at the point of its generation.
	 */
	public IntToStringDecoder generateDecoder() {
		String[][] nodeIdsForGraphsDecoder = new String[graphIndexedNodeIdEncoders.size()][];
		for (int i = 0; i < nodeIdsForGraphsDecoder.length; i++) {
			UniqueIntEncoder<String> graphIndexedNodeIdEncoder = graphIndexedNodeIdEncoders.get(i);
			nodeIdsForGraphsDecoder[i] = stringListToArray(graphIndexedNodeIdEncoder.getDecoderArrayList());
		}
		return new IntToStringDecoder(
				stringListToArray(graphIdEncoder.getDecoderArrayList()),
				stringListToArray(nodeLabelEncoder.getDecoderArrayList()),
				stringListToArray(edgeLabelEncoder.getDecoderArrayList()),
				nodeIdsForGraphsDecoder);
	}

	private String[] stringListToArray(List<String> list) {
		return list.toArray(new String[list.size()]);
	}
	
}
