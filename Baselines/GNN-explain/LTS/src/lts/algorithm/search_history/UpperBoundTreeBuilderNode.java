package lts.algorithm.search_history;

import java.util.TreeMap;

public class UpperBoundTreeBuilderNode {
		
	private final TreeMap<Integer, UpperBoundTreeBuilderNode> map = new TreeMap<>();

	protected UpperBoundTreeBuilderNode addChild(int bin) {
		UpperBoundTreeBuilderNode child = map.get(bin);
		if (child == null) {
			map.put(bin, child = new UpperBoundTreeBuilderNode());
		} 
		return child;
	}
	
	protected TreeMap<Integer, UpperBoundTreeBuilderNode> getChildren() {
		return map;
	}

}
