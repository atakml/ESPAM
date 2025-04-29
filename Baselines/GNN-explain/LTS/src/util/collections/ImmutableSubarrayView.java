package util.collections;

import java.util.Iterator;

public class ImmutableSubarrayView<T> implements ImmutableArrayView<T> {

	private final class ImmutableSubarrayView_Iterator implements Iterator<T> {
		
		int curr = 0;
		
		@Override
		public boolean hasNext() {
			return curr != ImmutableSubarrayView.this.len;
		}

		@Override
		public T next() {
			return ImmutableSubarrayView.this.data[ImmutableSubarrayView.this.start + curr++];
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException();
		}
		
	}
	
	private final T[] data;
	private final int start, len;
	
	public ImmutableSubarrayView(T[] data, int start, int len) {
		this.data = data;
		this.start = start;
		this.len = len;
	}
	
	@Override
	public Iterator<T> iterator() {
		return new ImmutableSubarrayView_Iterator();
	}

	@Override
	public T get(int index) throws IndexOutOfBoundsException {
		checkIndex(index);
		return data[start + index];
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
