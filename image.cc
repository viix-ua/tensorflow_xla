
#include "image.h"
#include "reference_util.h"

namespace xla
{

   image_file_type::type read_type(const std::string& file_name)
   {
      std::ifstream file(file_name.c_str(), std::ios::in | std::ios::binary);
      if (!file)
      {
         throw image_load_error("Unable to open file: " + file_name);
      }

      char buffer[9];
      file.read((char*)buffer, 8);
      buffer[8] = 0;

      // Determine the true image type using link:
      // http://en.wikipedia.org/wiki/List_of_file_signatures

      if (strcmp(buffer, "\x89\x50\x4E\x47\x0D\x0A\x1A\x0A") == 0)
      {
         return image_file_type::PNG;
      }
      else if ((buffer[0] == '\xff') && (buffer[1] == '\xd8') && (buffer[2] == '\xff'))
      {
         return image_file_type::JPG;
      }
      else if ((buffer[0] == 'B') && (buffer[1] == 'M'))
      {
         return image_file_type::BMP;
      }
      else if ((buffer[0] == 'D') && (buffer[1] == 'N') && (buffer[2] == 'G'))
      {
         return image_file_type::DNG;
      }

      return image_file_type::UNKNOWN;
   }


   std::unique_ptr<xla::Array4D<UChar8>> load_image(const std::string& file_name)
   {
      const image_file_type::type im_type = read_type(file_name);

      if (im_type == image_file_type::BMP)
      {
         return load_bmp(file_name);
      }
      else
      {
         if (im_type == image_file_type::JPG)
            throw image_load_error("DLIB_JPEG_SUPPORT not #defined: Unable to load image in file " + file_name);
         else if (im_type == image_file_type::PNG)
            throw image_load_error("DLIB_PNG_SUPPORT not #defined: Unable to load image in file " + file_name);
         else
            throw image_load_error("Unknown image file format: Unable to load image in file " + file_name);
      }
   }


   void apply_img_filter(const xla::Array4D<xla::UChar8>& original, const std::string& filename, const xla::Array2D<float>& kernel)
   {
      auto img = xla::MakeUnique<xla::Array4D<xla::UChar8>>(3, 1, original.height(), original.width(), original.flatten());
      std::unique_ptr<xla::Array4D<float>>& img_float = img->convert<float>();

      auto img_emboss_float = xla::ReferenceUtil::Conv2D<float>(*img_float, kernel, { 1, 1 }, xla::Padding::kSame);

      xla::write_as_bmp(*img_emboss_float, filename);
   }


   std::unique_ptr<Array4D<float>> crop_image(const std::string& file_name, int64 rect_sz)
   {
      auto img = load_image(file_name);

      const int64 width = img->width();
      const int64 height = img->height();

      const int64 short_edge = std::min(width, height);
      if (rect_sz < short_edge)
      {
         auto crop = MakeUnique<Array4D<float>>(1, 3, short_edge, short_edge);

         int64 xx = int64((width - short_edge) / 2);
         int64 yy = int64((height - short_edge) / 2);

         for (int64 i = 0; i < short_edge; i++)
         {
            for (int64 j = 0; j < short_edge; j++)
            {
               (*crop)(0, 0, j, i) = (*img)(0, 0, j + yy, i + xx);
               (*crop)(0, 1, j, i) = (*img)(0, 1, j + yy, i + xx);
               (*crop)(0, 2, j, i) = (*img)(0, 2, j + yy, i + xx);
            }
         }
         auto resized_img = resizeBilinear(*crop, rect_sz, rect_sz);
         
         return resized_img;
      }
      else
      {
         return std::unique_ptr<Array4D<float>>(nullptr);
      }
   }


   std::unique_ptr<Array4D<float>> resizeBilinear(const xla::Array4D<float>& img, int64 new_width, int64 new_height)
   {
      auto resized_img = MakeUnique<xla::Array4D<float>>(1, 3, new_height, new_width);

      int x = 0, y = 0;

      const float x_ratio = ((float)(img.width() - 1)) / new_width;
      const float y_ratio = ((float)(img.height() - 1)) / new_height;

      float x_diff = 0.f, y_diff = 0.f;
      float blue = 0.f, red = 0.f, green = 0.f;

      for (int i = 0; i < new_height; i++)
      {
         for (int j = 0; j < new_width; j++)
         {
            x = (int)(x_ratio * j);
            y = (int)(y_ratio * i);

            x_diff = (x_ratio * j) - x;
            y_diff = (y_ratio * i) - y;

            const float a_r = img(0, 0, y, x); 
            const float a_g = img(0, 1, y, x);
            const float a_b = img(0, 2, y, x);

            const float b_r = img(0, 0, y, x + 1);
            const float b_g = img(0, 1, y, x + 1);
            const float b_b = img(0, 2, y, x + 1);

            const float c_r = img(0, 0, y + 1, x);
            const float c_g = img(0, 1, y + 1, x);
            const float c_b = img(0, 2, y + 1, x);

            const float d_r = img(0, 0, y + 1, x + 1);
            const float d_g = img(0, 1, y + 1, x + 1);
            const float d_b = img(0, 2, y + 1, x + 1);

            // blue element
            // Yb = Ab(1-w)(1-h) + Bb(w)(1-h) + Cb(h)(1-w) + Db(wh)
            blue = a_b*(1 - x_diff)*(1 - y_diff) + b_b*(x_diff)*(1 - y_diff) + c_b*(y_diff)*(1 - x_diff) + d_b*(x_diff*y_diff);

            // green element
            // Yg = Ag(1-w)(1-h) + Bg(w)(1-h) + Cg(h)(1-w) + Dg(wh)
            green = a_g*(1 - x_diff)*(1 - y_diff) + b_g*(x_diff)*(1 - y_diff) + c_g*(y_diff)*(1 - x_diff) + d_g*(x_diff*y_diff);

            // red element
            // Yr = Ar(1-w)(1-h) + Br(w)(1-h) + Cr(h)(1-w) + Dr(wh)
            red = a_r*(1 - x_diff)*(1 - y_diff) + b_r*(x_diff)*(1 - y_diff) + c_r*(y_diff)*(1 - x_diff) + d_r*(x_diff*y_diff);

            (*resized_img)(0, 0, i, j) = red;
            (*resized_img)(0, 1, i, j) = green;
            (*resized_img)(0, 2, i, j) = blue;
         }
      }
      return resized_img;
   }

}
