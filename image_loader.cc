
#include "image_loader.h"

namespace xla
{
   std::unique_ptr<xla::Array4D <UChar8> > load_bmp(const std::string& file_name)
   {
      std::ifstream fin(file_name.c_str(), std::ios::in | std::ios::binary);
      if (!fin)
      {
         //throw image_load_error("Unable to open " + file_name + " for reading.");
         return std::unique_ptr<xla::Array4D<UChar8>>();
      }

      unsigned long bytes_read_so_far = 0;

      using namespace std;

      streambuf& in = *fin.rdbuf();
      //        streamsize num;
      UChar8 buf[100];

      std::unique_ptr<xla::Array4D<UChar8>> image = 0;

      try
      {
         BITMAPFILEHEADER header;

         header.bfType = read_u16(in);
         header.bfSize = read_u32(in);
         header.bfReserved1 = read_u16(in);
         header.bfReserved2 = read_u16(in);
         header.bfOffBits = read_u32(in);

         buf[0] = UChar8(header.bfType);
         buf[1] = UChar8(header.bfType >> 8);


         bytes_read_so_far += 2;

         if (buf[0] != 'B' || buf[1] != 'M')
         {
            throw image_load_error("bmp load error 2: header error");
         }

         bytes_read_so_far += 12;
         // finish read BITMAPFILEHEADER

         // https://ziggi.org/bystryy-negativ-bmp-izobrazheniya-v-cpp/


         // if this value isn't zero then there is something wrong
         // with this bitmap.
         if (header.bfReserved1 != 0)
         {
            throw image_load_error("bmp load error 4: reserved area not zero");
         }


         bytes_read_so_far += 40;

         ///////////////////////////////////
         BITMAPINFOHEADER info;
         info.biSize = read_u32(in);
         info.biWidth = read_u32(in);
         info.biHeight = read_u32(in);
         info.biPlanes = read_u16(in);
         info.biBitCount = read_u16(in);
         info.biCompression = read_u32(in);
         info.biSizeImage = read_u32(in);
         info.biXPelsPerMeter = read_u32(in);
         info.biYPelsPerMeter = read_u32(in);
         info.biClrUsed = read_u32(in);
         info.biClrImportant = read_u32(in);

         image = xla::MakeUnique<xla::Array4D<UChar8>>(1, 3, info.biHeight, info.biWidth);

         switch (info.biBitCount)
         {
         case 1:
         {
            // figure out how the pixels are packed
            long padding;
            if (header.bfSize - header.bfOffBits == static_cast<unsigned int>(info.biWidth*info.biHeight) / 8U)
            {
               padding = 0;
            }
            else
            {
               padding = 4 - ((info.biWidth + 7) / 8) % 4;
            }

            const unsigned int palette_size = 2;
            UChar8 red[palette_size];
            UChar8 green[palette_size];
            UChar8 blue[palette_size];

            for (unsigned int i = 0; i < palette_size; ++i)
            {
               if (in.sgetn(reinterpret_cast<char*>(buf), 4) != 4)
               {
                  throw image_load_error("bmp load error 20: color palette missing");
               }
               bytes_read_so_far += 4;
               blue[i] = buf[0];
               green[i] = buf[1];
               red[i] = buf[2];
            }


            // seek to the start of the pixel data
            while (bytes_read_so_far != header.bfOffBits)
            {
               const long to_read = (long)std::min(header.bfOffBits - bytes_read_so_far, (unsigned long)sizeof(buf));
               if (in.sgetn(reinterpret_cast<char*>(buf), to_read) != to_read)
               {
                  throw image_load_error("bmp load error: missing data");
               }
               bytes_read_so_far += to_read;
            }

            // load the image data
            for (int row = info.biHeight - 1; row >= 0; --row)
            {
               for (int col = 0; col < info.biWidth; col += 8)
               {
                  if (in.sgetn(reinterpret_cast<char*>(buf), 1) != 1)
                  {
                     throw image_load_error("bmp load error 21.6: file too short");
                  }

                  UChar8 pixels[8];

                  pixels[0] = (buf[0] >> 7);
                  pixels[1] = ((buf[0] >> 6) & 0x01);
                  pixels[2] = ((buf[0] >> 5) & 0x01);
                  pixels[3] = ((buf[0] >> 4) & 0x01);
                  pixels[4] = ((buf[0] >> 3) & 0x01);
                  pixels[5] = ((buf[0] >> 2) & 0x01);
                  pixels[6] = ((buf[0] >> 1) & 0x01);
                  pixels[7] = ((buf[0]) & 0x01);

                  for (int i = 0; i < 8 && col + i < info.biWidth; ++i)
                  {
                     rgb_pixel p;
                     p.red = red[pixels[i]];
                     p.green = green[pixels[i]];
                     p.blue = blue[pixels[i]];

                     (*image)(0, 0, row, col + i) = p.red;
                     (*image)(0, 1, row, col + i) = p.green;
                     (*image)(0, 2, row, col + i) = p.blue;
                  }
               }
               if (in.sgetn(reinterpret_cast<char*>(buf), padding) != padding)
               {
                  throw image_load_error("bmp load error 9: file too short");
               }
            }
         }
         break;
         case 4:
         {
            // figure out how the pixels are packed
            long padding;
            if (header.bfSize - header.bfOffBits == static_cast<unsigned int>(info.biWidth*info.biHeight) / 2U)
            {
               padding = 0;
            }
            else
            {
               padding = 4 - ((info.biWidth + 1) / 2) % 4;
            }

            const int palette_size = 16;
            UChar8 red[palette_size];
            UChar8 green[palette_size];
            UChar8 blue[palette_size];

            for (int i = 0; i < palette_size; ++i)
            {
               if (in.sgetn(reinterpret_cast<char*>(buf), 4) != 4)
               {
                  throw image_load_error("bmp load error 20: color palette missing");
               }
               bytes_read_so_far += 4;
               blue[i] = buf[0];
               green[i] = buf[1];
               red[i] = buf[2];
            }

            // seek to the start of the pixel data
            while (bytes_read_so_far != header.bfOffBits)
            {
               const long to_read = (long)std::min(header.bfOffBits - bytes_read_so_far, (unsigned long)sizeof(buf));
               if (in.sgetn(reinterpret_cast<char*>(buf), to_read) != to_read)
               {
                  throw image_load_error("bmp load error: missing data");
               }
               bytes_read_so_far += to_read;
            }

            // load the image data
            for (int row = info.biHeight - 1; row >= 0; --row)
            {
               for (int col = 0; col < info.biWidth; col += 2)
               {
                  if (in.sgetn(reinterpret_cast<char*>(buf), 1) != 1)
                  {
                     throw image_load_error("bmp load error 21.7: file too short");
                  }

                  const unsigned char pixel1 = (buf[0] >> 4);
                  const unsigned char pixel2 = (buf[0] & 0x0F);

                  rgb_pixel p;
                  p.red = red[pixel1];
                  p.green = green[pixel1];
                  p.blue = blue[pixel1];

                  (*image)(0, 0, row, col) = p.red;
                  (*image)(0, 1, row, col) = p.green;
                  (*image)(0, 2, row, col) = p.blue;

                  if (col + 1 < info.biWidth)
                  {
                     p.red = red[pixel2];
                     p.green = green[pixel2];
                     p.blue = blue[pixel2];
                     //(*image)(row, col+1) = p;
                     (*image)(0, 0, row, col + 1) = p.red;
                     (*image)(0, 1, row, col + 1) = p.green;
                     (*image)(0, 2, row, col + 1) = p.blue;
                  }
               }
               if (in.sgetn(reinterpret_cast<char*>(buf), padding) != padding)
                  throw image_load_error("bmp load error 9: file too short");
            }
         }
         break;
         case 8:
         {
            // figure out how the pixels are packed
            int padding = 0;
            if (header.bfSize - header.bfOffBits == static_cast<unsigned int>(info.biWidth*info.biHeight))
               padding = 0;
            else
               padding = 4 - info.biWidth % 4;

            // check for this case.  It shouldn't happen but some BMP writers screw up the files
            // so we have to do this.
            if (info.biHeight * (info.biWidth + padding) > static_cast<int>(header.bfSize - header.bfOffBits))
               padding = 0;

            const unsigned int palette_size = 256;
            UChar8 red[palette_size];
            UChar8 green[palette_size];
            UChar8 blue[palette_size];

            for (unsigned int i = 0; i < palette_size; ++i)
            {
               if (in.sgetn(reinterpret_cast<char*>(buf), 4) != 4)
               {
                  throw image_load_error("bmp load error 20: color palette missing");
               }
               bytes_read_so_far += 4;

               blue[i] = buf[0];
               green[i] = buf[1];
               red[i] = buf[2];
            }


            // seek to the start of the pixel data
            while (bytes_read_so_far != header.bfOffBits)
            {
               const long to_read = (long)std::min(header.bfOffBits - bytes_read_so_far, (unsigned long)sizeof(buf));
               if (in.sgetn(reinterpret_cast<char*>(buf), to_read) != to_read)
               {
                  throw image_load_error("bmp load error: missing data");
               }
               bytes_read_so_far += to_read;
            }

            // Next we load the image data.

            // if there is no RLE compression
            if (info.biCompression == 0)
            {
               for (long row = info.biHeight - 1; row >= 0; --row)
               {
                  for (/*unsigned long*/ int col = 0; col < info.biWidth; ++col)
                  {
                     if (in.sgetn(reinterpret_cast<char*>(buf), 1) != 1)
                     {
                        throw image_load_error("bmp load error 21.8: file too short");
                     }

                     rgb_pixel p;
                     p.red = red[buf[0]];
                     p.green = green[buf[0]];
                     p.blue = blue[buf[0]];

                     //(*image)(row, col) = p;
                     (*image)(0, 0, row, col) = p.red;
                     (*image)(0, 1, row, col) = p.green;
                     (*image)(0, 2, row, col) = p.blue;
                  }

                  if (in.sgetn(reinterpret_cast<char*>(buf), padding) != padding)
                  {
                     throw image_load_error("bmp load error 9: file too short");
                  }
               }
            }
            else
            {
               // Here we deal with the psychotic RLE used by BMP files.

               // First zero the image since the RLE sometimes jumps over
               // pixels and assumes the image has been zero initialized.
               //assign_all_pixels(image, 0);

               long row = info.biHeight - 1;
               long col = 0;
               while (true)
               {
                  if (in.sgetn(reinterpret_cast<char*>(buf), 2) != 2)
                  {
                     throw image_load_error("bmp load error 21.9: file too short");
                  }

                  const unsigned char count = buf[0];
                  const unsigned char command = buf[1];

                  if (count == 0 && command == 0)
                  {
                     // This is an escape code that means go to the next row
                     // of the image
                     --row;
                     col = 0;
                     continue;
                  }
                  else if (count == 0 && command == 1)
                  {
                     // This is the end of the image.  So quit this loop.
                     break;
                  }
                  else if (count == 0 && command == 2)
                  {
                     // This is the escape code for the command to jump to
                     // a new part of the image relative to where we are now.
                     if (in.sgetn(reinterpret_cast<char*>(buf), 2) != 2)
                     {
                        throw image_load_error("bmp load error 21.1: file too short");
                     }
                     col += buf[0];
                     row -= buf[1];
                     continue;
                  }
                  else if (count == 0)
                  {
                     // This is the escape code for a run of uncompressed bytes

                     if (row < 0 || col + command > image->width())
                     {
                        // If this is just some padding bytes at the end then ignore them
                        if (row >= 0 && col + count <= image->width() + padding)
                           continue;

                        throw image_load_error("bmp load error 21.2: file data corrupt");
                     }

                     // put the bytes into the image
                     for (unsigned int i = 0; i < command; ++i)
                     {
                        if (in.sgetn(reinterpret_cast<char*>(buf), 1) != 1)
                        {
                           throw image_load_error("bmp load error 21.3: file too short");
                        }
                        rgb_pixel p;
                        p.red = red[buf[0]];
                        p.green = green[buf[0]];
                        p.blue = blue[buf[0]];

                        (*image)(0, 0, row, col) = p.red;
                        (*image)(0, 1, row, col) = p.green;
                        (*image)(0, 2, row, col) = p.blue;

                        ++col;
                     }

                     // if we read an uneven number of bytes then we need to read and
                     // discard the next byte.
                     if ((command & 1) != 1)
                     {
                        if (in.sgetn(reinterpret_cast<char*>(buf), 1) != 1)
                        {
                           throw image_load_error("bmp load error 21.4: file too short");
                        }
                     }

                     continue;
                  }

                  rgb_pixel p;

                  if (row < 0 || col + count > image->width())
                  {
                     // If this is just some padding bytes at the end then ignore them
                     if (row >= 0 && col + count <= image->width() + padding)
                     {
                        continue;
                     }

                     throw image_load_error("bmp load error 21.5: file data corrupt");
                  }

                  // put the bytes into the image
                  for (unsigned int i = 0; i < count; ++i)
                  {
                     p.red = red[command];
                     p.green = green[command];
                     p.blue = blue[command];

                     (*image)(0, 0, row, col) = p.red;
                     (*image)(0, 1, row, col) = p.green;
                     (*image)(0, 2, row, col) = p.blue;

                     ++col;
                  }
               }
            }
         }
         break;
         case 16:
            throw image_load_error("16 bit BMP images not supported");
         case 24:   //
         {
            // figure out how the pixels are packed
            long padding;
            if (header.bfSize - header.bfOffBits == static_cast<unsigned int>(info.biWidth * info.biHeight) * 3U)
            {
               padding = 0;
            }
            else
            {
               padding = 4 - (info.biWidth * 3) % 4;
            }

            // check for this case.  It shouldn't happen but some BMP writers screw up the files
            // so we have to do this.
            if (info.biHeight * (info.biWidth * 3 + padding) > static_cast<int>(header.bfSize - header.bfOffBits))
            {
               padding = 0;
            }

            // seek to the start of the pixel data
            while (bytes_read_so_far != header.bfOffBits)
            {
               const long to_read = (long)std::min(header.bfOffBits - bytes_read_so_far, (unsigned long)sizeof(buf));
               if (in.sgetn(reinterpret_cast<char*>(buf), to_read) != to_read)
               {
                  throw image_load_error("bmp load error: missing data");
               }
               bytes_read_so_far += to_read;
            }

            // load the image data
            for (int row = info.biHeight - 1; row >= 0; --row)
            {
               for (int col = 0; col < info.biWidth; ++col)
               {
                  if (in.sgetn(reinterpret_cast<char*>(buf), 3) != 3)
                  {
                     throw image_load_error("bmp load error 8: file too short");
                  }

                  rgb_pixel p;
                  p.red = buf[2];
                  p.green = buf[1];
                  p.blue = buf[0];

                  (*image)(0, 2, row, col) = p.red;
                  (*image)(0, 1, row, col) = p.green;
                  (*image)(0, 0, row, col) = p.blue;

               }

               if (padding > 0)
               {
                  if (in.sgetn(reinterpret_cast<char*>(buf), padding) != padding)
                  {
                     throw image_load_error("bmp load error 9: file too short");
                  }
               }
            }

            break;
         }
         case 32:
            throw image_load_error("32 bit BMP images not supported");
         default:
            throw image_load_error("bmp load error 10: unknown color depth");

         }
      }
      catch (...)
      {
         //image.clear();
         throw;
      }
      return image;
   }

}  // ns
