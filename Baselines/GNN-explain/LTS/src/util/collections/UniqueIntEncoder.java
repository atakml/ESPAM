package util.collections;

import java.util.ArrayList;
import java.util.Map;

/**
 * Given an set of items to encode, possibly with duplicates, an integer is assigned to each unique item(s),
 * in the range of 0 to N-1 where there are N unique items.
 * @author Robert
 *
 * @param <T>
 */
public class UniqueIntEncoder<T> {

	private final Map<T, Integer> map; 
	
	public UniqueIntEncoder() {
		this(MapType.HASH);
	}
	
	public UniqueIntEncoder(MapType mapType) {
		map = mapType.newMap();
	}
	
	public UniqueIntEncoder(Map<T, Integer> backingMap) {
		map = backingMap;
	}
	
	public int encode(T item) {
		Integer code = map.get(item);
		if (code == null) {
			code = map.size();
			if (code == Integer.MAX_VALUE) throw new RuntimeException("Items encoded exceeds max value of int.");
			map.put(item, code);
		}
		return code;
	}
	
	public int verify(T item) {
		Integer code = map.get(item);
		return code == null ? -1 : code;
	}
	
	public int size() {
		return map.size();
	}
	
	/**Reverses encoding : given an int X from the encoder, the object at index X is the object that produced the code*/
	public ArrayList<T> getDecoderArrayList() {
		ArrayList<T> decoder = CollectionUtil.nSizeArrayList(map.size(), null);
		for (Map.Entry<T, Integer> entry : map.entrySet()) {
			decoder.set(entry.getValue(), entry.getKey());
		}
		return decoder;
	}
}
