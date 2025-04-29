package util.collections;

public interface ImmutableIntArrayView {
	public int get(int index) throws IndexOutOfBoundsException;	
	public int length();
}
