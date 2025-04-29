package util.collections;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**Highly compact, random access, non-extensible representation of a list of lists.
 */
public class CompactDenseListList<T> implements CompactListList<T> {
	
	private final int[] listOffsets;
	private final T[] data;
		
	@SuppressWarnings("unchecked")
	public CompactDenseListList(int[] listLengths) {
		listOffsets = new int[listLengths.length];
		int dataLen = 0;
		for (int i = 0; i < listLengths.length; i++) {
			listOffsets[i] = dataLen;
			int listLength = listLengths[i];
			if (listLength < 0) {
				throw new IllegalArgumentException("listLengths["+i+"] = "+listLength+" which is < 0");
			}
			dataLen += listLength;
		}
		data = (T[]) new Object[dataLen];
	}
	
	@SuppressWarnings("unchecked")
	public <L extends List<T>> CompactDenseListList(List<L> listOfLists) {
		if (listOfLists == null || listOfLists.isEmpty()) {
			listOffsets = null;
			data = null;
		} else {
			listOffsets = new int[listOfLists.size()];
			int dataLen = 0;
			for (List<T> intList : listOfLists) {
				dataLen += intList == null ? 0 : intList.size(); 
			}
			data = (T[]) new Object[dataLen];
						
			int listOffset = 0, dataIndex = 0;
			for (List<T> list : listOfLists) {
				listOffsets[listOffset++] = dataIndex;
				if (list != null) {
					for (T item : list) {
						data[dataIndex++] = item;
					}
				}
			}
		}
	}
	
	private void checkListIndex(int listIndex) throws IndexOutOfBoundsException {
		if (listIndex < 0 || listIndex >= numLists()) {
			throw new IndexOutOfBoundsException("listIndex@"+listIndex);
		}
	}
	
	private int checkIndexAndReturnStart(int listIndex, int index) throws IndexOutOfBoundsException {
		if (index < 0) {
			throw new IndexOutOfBoundsException("index@"+index);
		}
		int startIndex = listOffsets[listIndex];
		int endIndex = listIndex == listOffsets.length - 1 ? data.length : listOffsets[listIndex + 1];
		if (index >= (endIndex - startIndex)) {
			throw new IndexOutOfBoundsException("index@"+index);
		}
		return startIndex;
	}
	
	@Override
	public T get(int listIndex, int index) {
		checkListIndex(listIndex);
		int startIndex = checkIndexAndReturnStart(listIndex, index);
		return data[startIndex + index];
	}
	
	@Override
	public void set(int listIndex, int index, T value) {
		checkListIndex(listIndex);
		int startIndex = checkIndexAndReturnStart(listIndex, index);
		data[startIndex + index] = value;
	}
	
	@Override
	public ArrayView<T> getList(int listIndex) {
		checkListIndex(listIndex);
		int startIndex = listOffsets[listIndex];
		int endIndex = listIndex == listOffsets.length - 1 ? data.length : listOffsets[listIndex + 1];
		int len = endIndex - startIndex;
		return new SubarrayView<T>(data, startIndex, len);
	}

	@Override
	public int numLists() {
		return listOffsets == null ? 0 : listOffsets.length;
	}
	
	@Override
	public int listSize(int listIndex) {
		checkListIndex(listIndex);
		int startIndex = listOffsets[listIndex];
		int endIndex = listIndex == listOffsets.length - 1 ? data.length : listOffsets[listIndex + 1];
		return endIndex - startIndex;
	}
	
	// TESTING
	
	private static List<Integer> makeIntList(int... ints) {
		List<Integer> intList = new ArrayList<>(ints.length);
		for (int integer : ints) {
			intList.add(integer);
		}
		return intList;
	}
	
	public static void main(String[] args) {
		List<List<Integer>> intListList = new ArrayList<>();
		intListList.add(makeIntList());
		intListList.add(makeIntList(101,102));
		intListList.add(makeIntList(103));
		intListList.add(makeIntList(104,105,106));
		CompactDenseListList<Integer> cloil = new CompactDenseListList<>(intListList);
		System.out.println(Arrays.toString(cloil.listOffsets));
		System.out.println(Arrays.toString(cloil.data));
		System.out.println(cloil.get(2, 0));
		ArrayView<Integer> iav;
		
		iav = cloil.getList(3);
		for (int i = 0; i < iav.length(); i++) {
			System.out.print(iav.get(i)+", ");
		}
		System.out.println();
		cloil.set(3, 1, 1337);
		
		for (int i = 0; i < iav.length(); i++) {
			System.out.print(iav.get(i)+", ");
		}
		System.out.println();
	}
	
}
