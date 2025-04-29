package util.logging;

import java.io.PrintStream;

/**Logs stuff*/
public class Log {
	
	public static final int
	SILENT = 0,
	ERROR = 1,
	WARNING = 2,
	INFO = 3,
	VERBOSE = 4,
	DEBUG = 5;
	
	private static int logLevel = INFO;
	private static PrintStream outputStream = System.out;
	
	public static void setLogLevel(int logLevel) {
		Log.logLevel = logLevel;
	}
	
	public static void setOutputStream(PrintStream outputStream) {
		Log.outputStream = outputStream;
	}
	
	public static void error(String s) {
		if (logLevel >= ERROR) outputStream.println(s);
	}
	
	public static void warning(String s) {
		if (logLevel >= WARNING) outputStream.println(s);
	}
	
	public static void info(String s) {
		if (logLevel >= INFO) outputStream.println(s);
	}
	
	public static void verbose(String s) {
		if (logLevel >= VERBOSE) outputStream.println(s);
	}
	
	public static void debug(String s) {
		if (logLevel >= DEBUG) outputStream.println(s);
	}
	
}
