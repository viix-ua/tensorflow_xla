
#ifndef IMAGE_H
#define IMAGE_H

#include "image_loader.h"

namespace xla
{
   const xla::Array2D<float> kernel_emboss(
   {
      { -2.f, -1.f,  0.f },
      { -1.f,  1.f,  1.f },
      {  0.f,  1.f,  2.f }
   });

   const xla::Array2D<float> kernel_edge_excessively(
   {
      { 1.f,  1.f,  1.f },
      { 1.f, -7.f,  1.f },
      { 1.f,  1.f,  1.f }
   });

   const xla::Array2D<float> kernel_edges_blur(
   {
      { 0.f,  1.f,  0.f },
      { 1.f, -3.f,  1.f },
      { 0.f,  1.f,  0.f }
   });

   const xla::Array2D<float> kernel_prewitt_operator_modified(
   {
      { -1.f,  0.f,   1.f },
      { -1.f,  0.5f,  1.f },
      { -1.f,  0.f,   1.f }
   });

   //const xla::Array2D<float> kernel_edge_operator_modified(
   //{
   //   { -1.f,  0.f,   1.f },
   //   { -2.f,  0.5f,  2.f },
   //   { -1.f,  0.f,   1.f }
   //});

   //////////////////////////////////////////////////

   const xla::Array2D<float> kernel_enhanced_edge(
   {
      { -1.f,  -1.f,  -1.f },
      { -1.f,   9.f,  -1.f },
      { -1.f,  -1.f,  -1.f }
   });

   const xla::Array2D<float> kernel_medium_edge(
   {
      { -1.f,  -1.f,  -1.f },
      { -1.f,   8.f,  -1.f },
      { -1.f,  -1.f,  -1.f }
   });

   ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //apply_img_filter(*img, std::string("Ferrari_488_emboss.bmp"), xla::kernel_emboss);
   //apply_img_filter(*img, std::string("Ferrari_488_edge_excessively.bmp"), xla::kernel_edge_excessively);
   //apply_img_filter(*img, std::string("Ferrari_488_edges_blur.bmp"), xla::kernel_edges_blur);
   //apply_img_filter(*img, std::string("Ferrari_488_prewitt_operator.bmp"), xla::kernel_prewitt_operator_modified);

   void apply_img_filter(const xla::Array4D<xla::UChar8>& original, const std::string& filename, const xla::Array2D<float>& kernel);

   ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   namespace image_file_type
   {
      enum type
      {
         BMP,
         JPG,
         PNG,
         DNG,
         UNKNOWN
      };
   }


   image_file_type::type read_type(const std::string& file_name);

   std::unique_ptr<xla::Array4D<UChar8>> load_image(const std::string& file_name);

   /**
   * Bilinear resize RGB image.
   * pixels is an array of size w * h.
   * Target dimension is w2 * h2.
   *
   * @param img Image pixels.
   * @param w New width.
   * @param h New height.
   * @return New array with size w * h.
   */
   std::unique_ptr<Array4D<float>> resizeBilinear(const xla::Array4D<float>& img, int64 w, int64 h);

   std::unique_ptr<Array4D<float>> crop_image(const std::string& file_name, int64 rect_sz);


   template <typename TType> inline
   void write_as_bmp(const xla::Array4D<TType>& img, const std::string& filename)
   {
      // https://stackoverflow.com/questions/5420317/reading-and-writing-binary-file
      // https://gcc.gnu.org/onlinedocs/libstdc++/manual/streambufs.html

      std::ofstream ostr(filename.c_str(), std::ios::out | std::ios::binary);//std::fstream::binary
      if (ostr)
      {
         std::streambuf& pbuf = *ostr.rdbuf();

         BITMAPFILEHEADER fileheader;
         fileheader.bfSize = static_cast<unsigned int>(img.num_elements()) + fileheader.bfOffBits;

         BITMAPINFOHEADER infoheader;
         infoheader.biWidth = static_cast<int>(img.width());
         infoheader.biHeight = static_cast<int>(img.height());
         infoheader.biSizeImage = static_cast<int>(img.num_elements());

         write_header(fileheader, pbuf);
         write_info(infoheader, pbuf);

         for (int row = int(img.height()) - 1; row >= 0; --row)
         {
            for (int col = 0; col < int(img.width()); ++col)
            {
               if (img.size(0) == 3)
               {
                  for (size_t i = 0; i < 3; i++)
                  {
                     TType value = img(i, 0, row, col);
                     if (int(value) < 0)
                     {
                        value = TType(0);
                     }
                     if (int(value) > 255)
                     {
                        value = TType(255);
                     }
                     const UChar8 ch = static_cast<UChar8>(value);
                     pbuf.sputn(reinterpret_cast<const char*>(&ch), 1);
                  }
               }
               else if (img.size(1) == 3)
               {
                  for (size_t i = 0; i < 3; i++)
                  {
                     TType value = img(0, i, row, col);
                     if (int(value) < 0)
                     {
                        value = TType(0);
                     }
                     if (int(value) > 255)
                     {
                        value = TType(255);
                     }
                     const UChar8 ch = static_cast<UChar8>(value);
                     pbuf.sputn(reinterpret_cast<const char*>(&ch), 1);
                  }
               }
               else
               {
                  for (size_t i = 0; i < 3; i++)
                  {
                     TType value = img(0, 0, row, col);
                     if (int(value) < 0)
                     {
                        value = TType(0);
                     }
                     if (int(value) > 255)
                     {
                        value = TType(255);
                     }
                     const UChar8 ch = static_cast<UChar8>(value);
                     pbuf.sputn(reinterpret_cast<const char*>(&ch), 1);
                  }
               }
            }
         }
      }
   }

}

#endif // IMAGE_LOADE 

