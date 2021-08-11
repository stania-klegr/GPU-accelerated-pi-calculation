// OpenCL kernel 'throwDarts' using float type
__kernel void throwDarts(__global int *seeds, const int repeats, __global int *output){
	//declare vars
	int gid = get_global_id(0);
    int inside = 0;
    int rand = seeds[gid];
    float x = 0;
    float y = 0;

	for (int iter = 0; iter < repeats; iter++)
	{
		rand = 1103515245 * rand + 12345;//generates a new value for rand
        x = ((float) (rand & 0xFFFFFF)) / 0x1000000;//gets the value as a float

		rand = 1103515245 * rand + 12345;//generates a new value for rand
        y = ((float) (rand & 0xFFFFFF)) / 0x1000000;//gets the value as a float

        if (x * x + y * y < 1.0 ){
            // Dart (x, y) is inside the circle
        	inside++;
        }
	}
	//store out total in the array
	output[gid] = inside;
}

// OpenCL kernel 'throwDarts' using integer type
__kernel void throwDartsInt(__global int *seeds, const int repeats, __global int *output){
	//declare vars
    int gid = get_global_id(0);
    int inside = 0;
    int rand = seeds[gid];
    long x = 0;
    long y = 0;

	//NOTE I tried using ints for x and y but it seemed to be causing overflow

    for (int iter = 0; iter < repeats; iter++)
    {
        // TODO: write this code
        rand = 1103515245 * rand + 12345;//generates a new value for rand
        x = rand & 0xFFFFFF;//get the value as a long

        rand = 1103515245 * rand + 12345;//generates a new value for rand
        y = rand & 0xFFFFFF;//get the value as a long

        if (x * x + y * y < 0xFFFFFE000001){
            // Dart (x, y) is inside the circle
            inside++;
        }
    }
    //store out total in the array
    output[gid] = inside;
}

// Optional: OpenCL kernel 'throwDarts' using double type
// Implement this function only if your GPU supports double type.
__kernel void throwDartsDouble(__global int *seeds, const int repeats, __global int *output){
	// Your code goes here
}
