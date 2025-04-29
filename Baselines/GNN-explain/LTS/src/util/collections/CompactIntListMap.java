package util.collections;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

/**Highly compact, random access, non-extensible multimap, with keys as ints.
 */
public class CompactIntListMap<T> implements Iterable<IntListMapEntry<T>> {
	
	// First half of this is int keys, second half is offsets within data array.
	// Keys are sorted, allowing lookup via binary search.
	private final int[] dataIndex;
	private final T[] data;
	
	/**Copies the data in the given map. Keys that have null or empty Lists are ignored.
	 * @param map Map of Integers to Lists.
	 * @param sortedMap Whether or not the Map is naturally sorted by keys, like a TreeMap.
	 * If it is, there's no need to sort in the Constructor.
	 * I could have checked whether map was an instance of SortedMap but that might not be valid
	 * since they could be used a custom comparator for the keys.
	 */
	@SuppressWarnings("unchecked")
	public <L extends List<T>> CompactIntListMap(Map<Integer, L> map, boolean sortedMap) {
		int nonEmpties = 0;
		int total = 0;
		for (L list : map.values()) {
			if (list != null && !list.isEmpty()) {
				nonEmpties++;
				total += list.size();
			}
		}
		if (nonEmpties == 0) {
			dataIndex = null;
			data = null;
		} else {
			dataIndex = new int[nonEmpties * 2];
			data = (T[]) new Object[total];
			
			int indexC = 0, dataC = 0;
			Collection<Map.Entry<Integer, L>> sortedEntries = sortedMap ? map.entrySet() : sortMapByKey(map);
			for (Map.Entry<Integer, L> entry : sortedEntries) {
				int listIndex = entry.getKey();
				L list = entry.getValue();
				
				if (list != null && !list.isEmpty()) {
					dataIndex[indexC] = listIndex;
					dataIndex[indexC + nonEmpties] = dataC;
					indexC++;
					for (T item : list) {
						data[dataC++] = item;
					}
				}
			}

		}
		
	}

	private static Comparator<Map.Entry<Integer, ?>> compareEntryKey = new Comparator<Map.Entry<Integer, ?>>() {
		@Override
		public int compare(Entry<Integer, ?> o1, Entry<Integer, ?> o2) {
			return o1.getKey() - o2.getKey();
		}
	};
	
	private <L extends List<T>> List<Map.Entry<Integer, L>> sortMapByKey(Map<Integer, L> map) {
		List<Entry<Integer, L>> entries = new ArrayList<>(map.entrySet());
		Collections.sort(entries, compareEntryKey);
		return entries;
	}
	
	public T get(int listIndex, int index) {
		return data[getTrueIndex(listIndex, index)];
	}
	
	public T getFirst() {
		return data[0];
	}

	public void set(int listIndex, int index, T value) {
		data[getTrueIndex(listIndex, index)] = value;
	}
	
	private int getTrueIndex(int listIndex, int index) {
		int listIndexIndex = findListIndexIndex(listIndex);
		if (listIndexIndex < 0) throw new IndexOutOfBoundsException("listIndex@"+listIndex);
		int listOffsetIndex = dataIndex.length / 2 + listIndexIndex;
		int startIndex = dataIndex[listOffsetIndex];
		int endIndex = listOffsetIndex == dataIndex.length - 1 ? data.length : dataIndex[listOffsetIndex + 1];
		int trueIndex = startIndex + index;
		if (trueIndex >= endIndex) throw new IndexOutOfBoundsException("index@"+index);
		return trueIndex;
	}
	
	private int findListIndexIndex(int listIndex) {
		return Arrays.binarySearch(dataIndex, 0, dataIndex.length / 2, listIndex);
	}

	public ArrayView<T> getList(int listIndex) {
		int listIndexIndex = findListIndexIndex(listIndex);
		if (listIndexIndex < 0) throw new IndexOutOfBoundsException("listIndex@"+listIndex);
		int listOffsetIndex = dataIndex.length / 2 + listIndexIndex;
		int startIndex = dataIndex[listOffsetIndex];
		int endIndex = listOffsetIndex == dataIndex.length - 1 ? data.length : dataIndex[listOffsetIndex + 1];
		return new SubarrayView<T>(data, startIndex, endIndex - startIndex);
	}

	public int listSize(int listIndex) {
		int listIndexIndex = findListIndexIndex(listIndex);
		if (listIndexIndex < 0) {
			return 0;
		}
		int listOffsetIndex = dataIndex.length / 2 + listIndexIndex;
		int startIndex = dataIndex[listOffsetIndex];
		int endIndex = listOffsetIndex == dataIndex.length - 1 ? data.length : dataIndex[listOffsetIndex + 1];
		return endIndex - startIndex;
	}

	@Override
	public Iterator<IntListMapEntry<T>> iterator() {
		return new Iterator<IntListMapEntry<T>>() {

			int halfDataIndex = dataIndex.length / 2;
			int dataIndexC = 0;
			
			
			@Override
			public boolean hasNext() {
				return dataIndexC < halfDataIndex;
			}

			@Override
			public IntListMapEntry<T> next() {
				int listOffsetC = dataIndexC + halfDataIndex;
				int start = dataIndex[listOffsetC++];
				int end = listOffsetC == dataIndex.length ? data.length : dataIndex[listOffsetC];
				IntListMapEntry<T> entry = new IntListMapEntry<>(
						dataIndex[dataIndexC],
						new SubarrayView<>(data, start, end - start));
				dataIndexC++;
				return entry;
			}

			@Override
			public void remove() {
				throw new UnsupportedOperationException();
			}
			
		};
	}
	
	public IntArrayView getKeys() {
		return new IntSubarrayView(dataIndex, 0, dataIndex.length / 2);
	}
	
	/**Total number of values in this map*/
	public int size() {
		return data == null ? 0 : data.length;
	}
	
	// TESTING
	
	public String toString() {
		return Arrays.toString(dataIndex) + Arrays.toString(data);
	}
	
	private static List<Integer> makeIntList(int... ints) {
		List<Integer> intList = new ArrayList<>(ints.length);
		for (int integer : ints) {
			intList.add(integer);
		}
		return intList;
	}
	
	// DEBUG
	public static void main(String[] args) {
		Map<Integer, List<Integer>> intListList = new HashMap<>();
		intListList.put(1,makeIntList());
		intListList.put(2,makeIntList(101,102));
		intListList.put(3,makeIntList(103));
		intListList.put(4,makeIntList(104,105,106));
		CompactIntListMap<Integer> cloil = new CompactIntListMap<>(intListList, false);
		System.out.println(Arrays.toString(cloil.dataIndex));
		System.out.println(Arrays.toString(cloil.data));
		System.out.println(cloil.get(2, 0));
	}
	
}
