
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
   // Represent notMNIST datafiles as C++ file streams f1 and f2 respectively
   ifstream f1("../../trainDataset-20k-notMNIST.bin", ios::in | ios::binary); // image data
   ifstream f2("../../trainLabels-20k-notMNIST.bin",  ios::in | ios::binary); // label data

   if (!f1.is_open() || !f2.is_open())
   {
      cerr << "ERROR: Can't open MNIST files. Please locate them in current directory" << endl;
      return 1;
   }

   try
   {
      // Here is our info
      const int imgno = 20000;  // images in file
      const int imgheight = 28; // image size
      const int imgwidth = 28;

      // Initialize error counter
      //int errors = 0;

      std::vector<float> data(imgheight * imgwidth);
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

   return 0;
}
