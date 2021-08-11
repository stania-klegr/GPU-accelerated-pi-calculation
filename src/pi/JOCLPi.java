package pi;

import static org.jocl.CL.CL_CONTEXT_PLATFORM;
import static org.jocl.CL.CL_DEVICE_NAME;
import static org.jocl.CL.CL_MEM_COPY_HOST_PTR;
import static org.jocl.CL.CL_MEM_READ_ONLY;
import static org.jocl.CL.CL_MEM_READ_WRITE;
import static org.jocl.CL.CL_PLATFORM_NAME;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clBuildProgram;
import static org.jocl.CL.clCreateBuffer;
import static org.jocl.CL.clCreateCommandQueue;
import static org.jocl.CL.clCreateContext;
import static org.jocl.CL.clCreateKernel;
import static org.jocl.CL.clCreateProgramWithSource;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clEnqueueReadBuffer;
import static org.jocl.CL.clReleaseCommandQueue;
import static org.jocl.CL.clReleaseContext;
import static org.jocl.CL.clReleaseKernel;
import static org.jocl.CL.clReleaseMemObject;
import static org.jocl.CL.clReleaseProgram;
import static org.jocl.CL.clSetKernelArg;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;

/**
 * Calculate the pi value using dart-throwing technique (Monte Carlo method).
 * 
 * This method is to generate a large number of random darts and check each
 * dart's distance is smaller than the radius. If so, then the dart is inside
 * the circle.
 * 
 * Then the ratio of darts inside the circle to the total number of dots is the
 * value of π/4.
 * 
 * 
 * Reference link:https://en.wikipedia.org/wiki/Pi
 */
public class JOCLPi {

	static int[] results;//for java version
	
	/**
	 * A dummy Java-version of our kernel. This is useful so that we can test and
	 * debug it in Java first.
	 * 
	 * @param seeds   one integer seed for each thread (work item).
	 * @param repeats the number of darts each thread must throw.
	 * @param output  one integer output cell for each thread
	 * @param gid     dummy global id, only needed in the Java API, not the OpenCL
	 *                version. (delete this parameter when you translate this to an
	 *                OpenCL kernel).
	 */
	public static void dummyThrowDarts(int[] seeds, int repeats, int[] output, int gid) {
		// int gid = get_global_id(0); // this is how we get the gid in OpenCL.
		
		results = output;
				
		//loop to start each of the threads
		for(int i = 0; i< seeds.length; i++){			
			piThread piemaker = new piThread(seeds[i], repeats, i);
			piemaker.start();
		}
		//loop to wait for the results the threads
		for(int i = 0; i< seeds.length; i++){
			//wait for the threads to all return results
			while(results[i] == 0)//NOTE, this might cause synchronization error
			{
				System.out.println("waiting " + i);
			}
		}
			
		int inside = 0;	
		int total = repeats * seeds.length;
		//sum up the results from all the threads
		for (int o = 0; o < seeds.length; o++) {
			inside += results[o];
		}
		//calculate pi and print it out
		final double pi = 4.0 * inside / total;
		System.out.println("Estimate PI = " + inside + "/" + total + " = " + pi);
	}
	//thread to calculate pi
	static class piThread extends Thread{

		int seed, repeats, id;
		
		//constructor
	    public piThread(int seed, int repeats, int gid) {    
	    	this.seed = seed;
	    	this.repeats = repeats;
	    	this.id = gid;
	    }    
     
	    //the main method for the thread
	    public void run()
	    {
			int rand = seed;
			int inside = 0;
			float x, y;
	    	//main loop to throw the darts
			for (int iter = 0; iter < repeats; iter++) 
			{
				rand = 1103515245 * rand + 12345;//generates a new value for rand
	            x = ((float) (rand & 0xFFFFFF)) / 0x1000000;//get the random float between 0 and 1
	            
				rand = 1103515245 * rand + 12345;//generates a new value for rand
	            y = ((float) (rand & 0xFFFFFF)) / 0x1000000;//get the random float between 0 and 1

	            if (x*x + y*y < 1.0 ){
	                // Dart (x, y) is inside the circle
	            	inside++;
	            }
			}
	    	results[id] = inside;
	    }     
	}

	/**
	 * main arguments: threads workgroupsize repeats
	 * 
	 * threads: the total number of threads (work-items) workgroupsize: workgroup
	 * size ('threads' must be a multiplier of 'workgroupsize') repeats: the total
	 * number of darts the each thread generates
	 * 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		if (args.length != 3) {
			System.err.println("Usage: pi threads workgroupsize repeats kernel");
			System.err.println("      (threads must be a multiple of workgroupsize)");
			System.exit(1);
		}
		final int threads = Integer.decode(args[0]);
		final int wgSize = Integer.decode(args[1]);
		final int repeats = Integer.decode(args[2]);

		
		final String srcCode = new String(JOCLUtil.readResourceToString("/pi/kernel.cl"));
		
		// Enable openCL exceptions, so that we can avoid duplicated error checking in
		// the remaining program.
		CL.setExceptionsEnabled(true);
		final int platformIndex = 0; // Platform index
		final long deviceType = CL.CL_DEVICE_TYPE_GPU; // Show GPU device type
		final int deviceIndex = 0; // Device number

		// Obtain all platform ids on this machine.
		cl_platform_id[] platforms = JOCLUtil.getAllPlatforms();
		cl_platform_id platform = platforms[platformIndex];// Get the selected platform
		System.out.println("Selected CLPlatform: " + JOCLUtil.getPlatformInfoString(platform, CL_PLATFORM_NAME));// Show platform

		// Get all devices on the selected 'platform'
		cl_device_id[] devices = JOCLUtil.getAllDevices(platform, deviceType);
		cl_device_id device = devices[deviceIndex]; // Get a single device id
		System.out.println("Selected CLDevice: " + JOCLUtil.getDeviceInfoString(device, CL_DEVICE_NAME) + "\nDevice Version:"
				+ JOCLUtil.getDeviceInfoString(device, CL.CL_DEVICE_VERSION));// Show device
		// Initialize the context properties
		cl_context_properties contextProperties = new cl_context_properties();
		contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
		// Create a context for the selected device with contextProperties
		cl_context context = clCreateContext(contextProperties, 1, new cl_device_id[] { device }, null, null, null);
		// Create a OpenCL 1.2 command-queue on the selected device with specified
		// context
		@SuppressWarnings("deprecation")
		cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, null); // OpenCL 1.2

		// Create input- and output array

		int seeds[] = new int[threads];// Array 'seeds' stores the seed number
		int output[] = new int[threads];// Array 'output' stores the total number of dart inside the circle

		// Fill array 'seeds' with 0 .. threads-1
		for (int i = 0; i < threads; i++) {
			seeds[i] = i; // seeds[0] = 0, seeds[1] = 1, seeds[2] = 2, ....
		}

		////////////////////////////////////////////////////////////////////////////////////////////////////
		// The below is the OpenCL-related code
		////////////////////////////////////////////////////////////////////////////////////////////////////
		Pointer ptrSeeds = Pointer.to(seeds);
		Pointer ptrOutput = Pointer.to(output);

		// Allocate OpenCL-hosted memory for inputs and output
		// Create a read-only memory on OpenCL device and copy the array 'seeds' from
		// host to device
		cl_mem memIn1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int * threads,
				ptrSeeds, null);
		// Create a read-write memory on OpenCL device (default value).
		cl_mem memOut = clCreateBuffer(context, CL_MEM_READ_WRITE, Sizeof.cl_int * threads, null, null);
		// Write the opencl kernel as a Java string

		// Load the source code 'srcCode' to the program object
		cl_program program = clCreateProgramWithSource(context, 1, new String[] { srcCode }, null, null);
		// Build the program
		clBuildProgram(program, 0, null, null, null, null);
		// Create the kernel
		cl_kernel kernel = clCreateKernel(program, "throwDarts", null);
		// Set the arguments for the kernel
		clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(memIn1)); // Array 'seeds'
		clSetKernelArg(kernel, 1, Sizeof.cl_int, Pointer.to(new int[] { repeats }));// 'repeat'
		clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(memOut));// Array 'output'

		// Set the work-item dimensions
		long global_work_size[] = new long[] { threads }; // Global work group size is the number of repeats
		long local_work_size[] = new long[] { wgSize };

		// Execute the kernel
		System.out.println("Starting with " + threads + " threads, each doing " + repeats + " repeats.");
		System.out.flush();

		// Start to measure the time
		final long time0 = System.nanoTime();
		// Start to execute the kernel with global and local workgroup size
		clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, global_work_size, local_work_size, 0, null, null);

		// Read and copy OpenCL 'memOut' array to host-allocated array 'output' that
		// 'ptrOutput' references to.
		clEnqueueReadBuffer(commandQueue, memOut, CL_TRUE, 0, threads * Sizeof.cl_int, ptrOutput, 0, null, null);

		// Release memory objects, kernel, program, queue and context
		clReleaseMemObject(memIn1);
		clReleaseMemObject(memOut);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(commandQueue);
		clReleaseContext(context);

		final long time1 = System.nanoTime();
		System.out.println("Done in " + (time1 - time0) / 1000 + " microseconds");// Get the elapsed time.
		////////////////////////////////////////////////////////////////////////////////////////////////////
		// End of OpenCL-related code
		////////////////////////////////////////////////////////////////////////////////////////////////////

		// Calculate PI
		long inside = 0;
		long total = (long) threads * repeats;
		for (int i = 0; i < threads; i++) {
//			System.out.println("[Thread]: thread " + i + " gives " + output[i]);
			inside += output[i];
		}
		final double pi = 4.0 * inside / total;
		System.out.println("Estimate PI = " + inside + "/" + total + " = " + pi);

	}

}
