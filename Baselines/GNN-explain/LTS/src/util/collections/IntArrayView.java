package util.collections;

public interface IntArrayView {

	public int get(int index) throws IndexOutOfBoundsException;

	public void set(int index, int value) throws IndexOutOfBoundsException;
	
	public int length();
	
}
