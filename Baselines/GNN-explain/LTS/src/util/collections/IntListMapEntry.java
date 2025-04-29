package util.collections;

/**An entry in CompactIntListMap.*/
public class IntListMapEntry<T> {
	
	private final int key;
	private final ArrayView<T> value;
	
	protected IntListMapEntry(int key, ArrayView<T> value) {
		this.key = key;
		this.value = value;
	}
	
	public int getKey() {
		return key;
	}
	
	public ArrayView<T> getValue() {
		return value;
	}
	
}
