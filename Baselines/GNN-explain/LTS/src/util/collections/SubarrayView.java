package util.collections;

import java.util.Iterator;

/**A view over a subsection of an array. 
 * This does not copy the parent array, so operations performed on this will reflect on the backing array.*/
public class SubarrayView<T> implements ArrayView<T> {

	private final class SubarrayView_Iterator implements Iterator<T> {
		
		int curr = 0;
		
		@Override
		public boolean hasNext() {
			return curr != SubarrayView.this.len;
		}

		@Override
		public T next() {
			return SubarrayView.this.data[SubarrayView.this.start + curr++];
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException();
		}
		
	}
	
	private final T[] data;
	private final int start, len;
	
	/**Creates a new SubarrayView starting at the given start index in the parent array and with the given length.*/
	public SubarrayView(T[] data, int start, int len) {
		if (start + len > data.length) throw new IndexOutOfBoundsException("Out of bounds of data array.");
		this.data = data;
		this.start = start;
		this.len = len;
	}
	
	@Override
	public Iterator<T> iterator() {
		return new SubarrayView_Iterator();
	}

	@Override
	public T get(int index) throws IndexOutOfBoundsException {
		checkIndex(index);
		return data[start + index];
	}

	@Override
	public void set(int index, T value) throws IndexOutOfBoundsException {
		checkIndex(index);
		data[start + index] = value;
	}

	@Override
	public int length() {
		return len;
	}

	private void checkIndex(int index) throws IndexOutOfBoundsException {
		if (index < 0 || index >= len) {
			throw new IndexOutOfBoundsException("index@"+index);
		}
	}
}
