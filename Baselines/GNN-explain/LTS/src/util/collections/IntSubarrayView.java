package util.collections;

public class IntSubarrayView implements IntArrayView {
	
	private final int[] array;
	private final int offset, length;
	
	public IntSubarrayView(int[] array, int offset, int length) {
		if (offset < 0 || length < 0) throw new IllegalArgumentException();
		this.array = array;
		this.offset = offset;
		this.length = length;
	}

	@Override
	public int get(int index) {
		return array[offset + index];
	}

	@Override
	public void set(int index, int value) {
		array[offset + index] = value;
	}

	@Override
	public int length() {
		return length;
	}

}
