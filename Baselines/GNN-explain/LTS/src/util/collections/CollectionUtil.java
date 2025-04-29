package util.collections;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

/**Provides some static utility functions for Collections.*/
public class CollectionUtil {
	
	public static <T> ArrayList<T> nSizeArrayList(int n, T fillItem) {
		ArrayList<T> list = new ArrayList<>(n);
		extendListToIndex(list, n, fillItem);
		return list;
	}
	
	/**Adds null elements to the list until there exists an element at the given index.*/
	public static void extendListToIndex(List<?> list, int index) {
		extendListToIndex(list, index, null);
	}
	
	/**Adds the given element to the list until there exists an element at the given index.*/
	public static <T> void extendListToIndex(List<T> list, int index, T fillItem) {
		while (list.size() <= index) {
			list.add(fillItem);
		}
	}
	
	/**Adds an item to a map of ArrayLists with the key.
	 * Creates an ArrayList at the key if one did not already exist.*/
	public static <K, V> void addItemToMapOfArrayLists(Map<K, ArrayList<V>> map, K key, V item) {
		ArrayList<V> list = map.get(key);
		if (list == null) {
			map.put(key, list = new ArrayList<>());
		}
		list.add(item);
	}
	
	/**Adds an item to a list of ArrayLists with the index.
	 * Creates an ArrayList at the index if one did not already exist.*/
	public static <T> void addItemToListOfArrayLists(List<ArrayList<T>> listList, int index, T item) {
		extendListToIndex(listList, index);
		ArrayList<T> list = listList.get(index);
		if (list == null) {
			listList.set(index, list = new ArrayList<>());
		}
		list.add(item);
	}
	
	/**Returns an int array of a Collection of Integers.*/
	public static int[] toIntArray(Collection<Integer> col) {
		int[] array = new int[col.size()];
		int idx = 0;
		for (Integer v : col) {
			array[idx++] = v;
		}
		return array;
	}
	
	@SuppressWarnings("rawtypes")
	private static Iterator emptyImmutableIterator = new Iterator() {

		@Override
		public boolean hasNext() {
			return false;
		}

		@Override
		public Object next() {
			throw new NoSuchElementException("This iterator is empty.");
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException("This iterator is immutable");
		}
		
	};
	
	@SuppressWarnings("rawtypes")
	private static abstract class EmptyArrayView implements ArrayView, ImmutableArrayView {}
	
	@SuppressWarnings("rawtypes")
	private static EmptyArrayView emptyArrayView = new EmptyArrayView() {

		@Override
		public Iterator iterator() {
			return emptyImmutableIterator;
		}

		@Override
		public Object get(int index) throws IndexOutOfBoundsException {
			throw new IndexOutOfBoundsException("index@"+index);
		}

		@Override
		public void set(int index, Object value) throws IndexOutOfBoundsException {
			throw new IndexOutOfBoundsException("index@"+index);
		}

		@Override
		public int length() {
			return 0;
		}
		
	};
	
	/**Returns an empty ArrayView.*/
	@SuppressWarnings("unchecked")
	public static <T> ArrayView<T> getEmptyArrayView() {
		return (ArrayView<T>) emptyArrayView;
	}
	
	/**Returns an empty ImmutableArrayView.*/
	@SuppressWarnings("unchecked")
	public static <T> ImmutableArrayView<T> getImmutableEmptyArrayView() {
		return (ImmutableArrayView<T>) emptyArrayView;
	}
	
}
