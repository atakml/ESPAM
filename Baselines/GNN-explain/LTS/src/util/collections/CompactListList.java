package util.collections;

public interface CompactListList<T> {

	public T get(int listIndex, int index);
	
	public void set(int listIndex, int index, T value);
	
	public ArrayView<T> getList(int listIndex);
	
	public int numLists();
	
	public int listSize(int listIndex);
	
}
