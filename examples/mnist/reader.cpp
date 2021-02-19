
#include <iostream>
#include <fstream>
#include <sstream>
#include <exception>

using namespace std;

typedef char byte_t;


int main()
{
	// Represent MNIST datafiles as C++ file streams f1 and f2 respectively
	ifstream f1("../../t10k-images-idx3-ubyte", ios::in | ios::binary); // image data
	ifstream f2("../../t10k-labels-idx1-ubyte", ios::in | ios::binary); // label data
	
	if (!f1.is_open() || !f2.is_open())
	{
		cerr << "ERROR: Can't open MNIST files. Please locate them in current directory" << endl;
		return 1;
	}

	// Create buffers for image data and correct labels
	const int BUF_SIZE = 2048;
   byte_t *buffer = new byte_t[BUF_SIZE]; 
   byte_t *label = new byte_t[2];

	// Block for catching file exceptions
	try
	{
		// Read headers
		f1.read(buffer, 16);
		f2.read(buffer, 8);
	
		// Here is our info
		const int imgno = 10000; // 10'000 images in file
		const int imgheight = 28; // image size
		const int imgwidth = 28;
		const int imgpadx = 2; // Pad images by 2 black pixels, so
		const int imgpady = 2; // the image becomes 32x32
		const int imgpaddedheight = imgheight + 2*imgpady; // padded image size
		const int imgpaddedwidth = imgwidth + 2*imgpadx;
		
		// Clean the buffer
		memset(buffer, 0, BUF_SIZE);
	
		// Initialize error counter
		//int errors = 0;
	
		// Now cycle over all images in MNIST test dataset
		for (int i = 0; i < imgno; i++)
		{
			// Load the image from file stream into img32
			for (int k = 0; k < imgheight; k++)
			{
				// Image in file is stored as 28x28, so we need to pad it to 32x32
				// So we read the image row-by-row with proper padding adjustments
				f1.read(&buffer[imgpadx + (imgpaddedwidth)*(k + 2)], imgwidth);
			}

         for (int k = 0; k < imgheight; k++)
         {
            for (int n = 0; n < imgwidth; n++)
            {
               int value = int(0) | unsigned char(buffer[imgpadx + (imgpaddedwidth)*(k + 2) + n]);
               value /= 32;
               if (value)
               {
                  std::cout << value << ',';
               }
               else
                  std::cout << " ,";
            }
            std::cout << std::endl;
         }

         std::cout << "**********************" << std::endl;
	
			// Now read the correct label from label file stream
			f2.read(label, 1);
	
			// Check if our prediction is correct
			//if ( label[0] != pos ) errors++;
		}
		
		// Print the error rate
		//cout << "Error rate: " << (double)100.0*errors/imgno << "%" << endl;
		
	}
   catch (exception &e)
	{
		cerr << "Exception: " << e.what() << endl;
	}


	// Don't forget to free the memory
	delete[] label;
	delete[] buffer;

	// That's it!
	return 0;
}
