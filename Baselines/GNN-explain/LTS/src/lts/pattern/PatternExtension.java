package lts.pattern;

/**Describes a pattern extension. Used as a key to for grouping EmbeddingEdges that create the same pattern together*/
public interface PatternExtension {

	public EdgeType getType();
	
	public int getEdgeLabel();
	
}
