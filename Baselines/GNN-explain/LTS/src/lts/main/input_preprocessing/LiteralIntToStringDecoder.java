package lts.main.input_preprocessing;


/**
 * Decodes ints to their literal Strings. I used this for debugging purposes, and you may find it handy as well.
 * @author Robert
 *
 */
public class LiteralIntToStringDecoder extends IntToStringDecoder {

	private static LiteralIntToStringDecoder instance;
	
	public static LiteralIntToStringDecoder instance() {
		return instance == null ? instance = new LiteralIntToStringDecoder() : instance;
	}
	
	private LiteralIntToStringDecoder() {
		super(null, null, null, null);
	}
	
	public String decodeGraphId(int graphIdInt) {
		return Integer.toString(graphIdInt);
	}

	public String decodeNodeLabel(int nodeLabelInt) {
		return Integer.toString(nodeLabelInt);
	}

	public String decodeEdgeLabel(int edgeLabelInt) {
		return Integer.toString(edgeLabelInt);
	}

	public String decodeNodeIdForGraphs(int graphIdInt, int nodeIdInt) {
		return Integer.toString(nodeIdInt);
	}

}
