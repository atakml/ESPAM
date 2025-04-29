package util.collections;


/**Provides some static utility functions for bit manipulation.*/
public class BitUtil {

	public static int setIntBit(int intVal, int bitOffset, boolean value) {
		if (value) {
			intVal |= (1 << bitOffset);
		} else {
			intVal &= (~1 << bitOffset);
		}
		return intVal;
	}
	
	public static void setIntArrayBit(int[] array, int arrayOffset, int bitOffset, boolean value) {
		array[arrayOffset] = setIntBit(array[arrayOffset], bitOffset, value);
	}
	
	public static void setIntArrayBit(int[] array, int bitOffset, boolean value) {
		int arrayOffset = bitOffset / 32;
		setIntArrayBit(array, arrayOffset, bitOffset, value);
	}
	
	public static boolean getIntBit(int intVal, int bitOffset) {
		return ((intVal >> bitOffset) & 1) == 1;
	}
	
	public static boolean getIntArrayBit(int[] array, int arrayOffset, int bitOffset) {
		return getIntBit(array[arrayOffset], bitOffset);
	}
	
	public static boolean getIntArrayBit(int[] array, int bitOffset) {
		int arrayOffset = bitOffset / 32;
		return getIntArrayBit(array, arrayOffset, bitOffset);
	}
		
}
