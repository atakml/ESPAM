package lts.algorithm.search_history;

import java.util.Arrays;

public class UpperBoundTreeNode {

	private final int upperBoundBin;
	private final int childrenIndex[];
	private final UpperBoundTreeNode[] children;
	
	protected UpperBoundTreeNode(int upperBoundBin, int[] childrenIndex, UpperBoundTreeNode[] children) {
		this.upperBoundBin = upperBoundBin;
		this.childrenIndex = childrenIndex;
		this.children = children;
	}
	
	protected int getUpperBoundBin() {
		return upperBoundBin;
	}
	
	protected UpperBoundTreeNode getChild(int bin) {
		int i = Arrays.binarySearch(childrenIndex, bin);
		if (i < 0) return null;
		else return children[i];
	}
	
	/** For debugging */
	@Override
    public String toString() {
    	StringBuilder sb = new StringBuilder();
    	buildString(sb, 0, 0);
    	return sb.toString();
    }
    
    private void buildString(StringBuilder sb, int bin, int shift) {
    	for (int i = 0; i < shift; i++) sb.append("| ");
    	sb.append(bin+" max:"+upperBoundBin+"\n");
    	for (int i = 0; i < childrenIndex.length; i++) {
    		children[i].buildString(sb, childrenIndex[i], shift + 1);
    	}
    }
	
}
