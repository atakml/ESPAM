package util.collections;

import java.util.HashMap;
import java.util.Map;

public enum MapType {
	HASH, TREE;
	
	public <K, V> Map<K, V> newMap() {
		switch (this) {
		case HASH: return new HashMap<K, V>();
		case TREE: return new HashMap<K, V>();
		default: throw new UnsupportedOperationException("Map type "+this.toString()+" is not supported");
		}
	}
	
}
