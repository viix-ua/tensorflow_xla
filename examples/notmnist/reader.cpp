
#include <iostream>
#include <fstream>
#include <sstream>
#include <exception>
#include <cstring>
#include <vector>

using namespace std;

typedef char byte_t;

int main()
{
   // Represent MNIST datafiles as C++ file streams f1 and f2 respectively
   ifstream f1("../../trainDataset-20k-notMNIST.bin", ios::in | ios::binary); // image data
   ifstream f2("../../trainLabels-20k-notMNIST.bin", ios::in | ios::binary);  // label data

   if (!f1.is_open() || !f2.is_open())
   {
      cerr << "ERROR: Can't open MNIST files. Please locate them in current directory" << endl;
      return 1;
   }

   const int BUF_SIZE = 2048;

   float *buffer = new float[BUF_SIZE];
   //byte_t *label = new byte_t[2];

   try
   {
      // Read headers
      //f1.read(buffer, 16);
      //f2.read(buffer, 8);

      // Here is our info
      const int imgno = 20000; // 10'000 images in file
      const int imgheight = 28; // image size
      const int imgwidth = 28;
      const int imgpadx = 2; // Pad images by 2 black pixels, so
                             //const int imgpady = 2; // the image becomes 32x32
                             //const int imgpaddedheight = imgheight + 2*imgpady; // padded image size
      const int imgpaddedwidth = imgwidth + 2 * imgpadx;

      // Clean the buffer
      memset(buffer, 0, BUF_SIZE);

      // Initialize error counter
      //int errors = 0;

      std::vector<float> data(28*28);
      int32_t label = 0;

      // Now cycle over all images in MNIST test dataset
      for (int i = 0; i < imgno; i++)
      {
         f2.read(reinterpret_cast<char*>(&label), sizeof(int32_t));

         // Load the image from file stream into img32
         for (int k = 0; k < imgheight; k++)
         {
            // Image in file is stored as 28x28, so we need to pad it to 32x32
            // So we read the image row-by-row with proper padding adjustments
            f1.read(reinterpret_cast<char*>(&data[imgwidth * k]), imgwidth * sizeof(float));
         }

         for (int k = 0; k < imgheight; k++)
         {
            for (int n = 0; n < imgwidth; n++)
            {
               // Re-range from signed char to unsigned char range(0..255).
               float value = data[imgwidth * k + n];

               if (value > 0.f)
               {
                  std::cout << 1 << ',';
               }
               else std::cout << " ,";
            }
            std::cout << std::endl;
         }

         // Now read the correct label from label file stream
         //f2.read(label, 1);

         std::cout << " ********************************" << std::endl;

         // Check if our prediction is correct
         //if (label[0] != pos) errors++;
      }

      //cout << "Error rate: " << (double)100.0*errors/imgno << "%" << endl;
   }
   catch (exception &e)
   {
      cerr << "Exception: " << e.what() << endl;
   }

   //delete[] label;
   delete[] buffer;

   return 0;
}
