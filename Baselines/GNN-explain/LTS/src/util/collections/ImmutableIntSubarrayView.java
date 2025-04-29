package util.collections;

public class ImmutableIntSubarrayView implements ImmutableIntArrayView {

	private final int[] array;
	private final int offset, length;
	
	public ImmutableIntSubarrayView(int[] array, int offset, int length) {
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
	public int length() {
		return length;
	}
	
}
