package util.collections;

import java.util.Comparator;

/**Provides some static utility functions for arrays.*/
public class ArraysUtil {

	/**Returns indices of the array that contain the minimum element. Ex: {2,1,3,1,1} returns {1,3,4}*/
	public static int[] getMinIndices(int[] array) {
		int minIndex = getMinIndex(array);
		if (minIndex == -1) return new int[0];
		int minVal = array[minIndex];
		
		int minC = 0;
		for (int i = 0; i < array.length; i++) {
			if (array[i] == minVal) minC++;
		}
		
		int j = 0;
		int[] minIndices = new int[minC];
		for (int i = 0; i < array.length; i++) {
			if (array[i] == minVal) minIndices[j++] = i;
		}
		
		return minIndices;
	}
	
	/**Returns the index of the smallest element in the array */
	public static int getMinIndex(int[] array) {
		int minVal = Integer.MAX_VALUE;
		int atIndex = -1;
		for (int i = 0; i < array.length; i++) {
			if (array[i] < minVal) {
				minVal = array[i];
				atIndex = i;
			}
		}
		
		return atIndex;
	}
	
	/**Lexicographic comparator*/
	public final static Comparator<int[]> LEX_COMPARATOR_INT_ARRAYS = new Comparator<int[]>() {

		@Override
		public int compare(int[] a, int[] b) {
			if (a == null) {
				if (b == null) {
					return 0;
				} else {
					return -1;  // a is "smaller" because it is empty
				}
			}
			
			if (b == null) {
				return 1;
			}
			
			int len = Math.min(a.length, b.length);
			for (int i = 0; i < len; i++) {
				if (a[i] != b[i]) return a[i] - b[i];
			}
			return a.length > b.length ? 1 : -1;
		}
		
	};
	
}
