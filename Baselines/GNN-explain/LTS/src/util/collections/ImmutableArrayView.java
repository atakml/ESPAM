package util.collections;

public interface ImmutableArrayView<T> extends Iterable<T> {

	public T get(int index) throws IndexOutOfBoundsException;
	
	public int length();
	
}
