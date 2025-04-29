package util.collections;

/**Provides the functionality of an array. */
public interface ArrayView<T> extends Iterable<T> {

	public T get(int index) throws IndexOutOfBoundsException;

	public void set(int index, T value) throws IndexOutOfBoundsException;
	
	/**Length of the array. */
	public int length();
	
}
