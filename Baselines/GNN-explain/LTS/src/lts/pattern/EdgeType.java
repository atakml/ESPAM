package lts.pattern;

/**Distinguishes between the initial base edge of a pattern,
 * Extensions from an existing node in the pattern to a new node,
 * and extensions of adding an edge between 2 existing nodes in the pattern.*/
public enum EdgeType {
	BASE, NEW_NODE, EXISTING_NODES;
}
