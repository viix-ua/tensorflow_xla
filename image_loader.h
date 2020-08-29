
#ifndef IMAGE_LOADER_BMP
#define IMAGE_LOADER_BMP


#include <iostream>
#include <sstream>
#include <algorithm>
#include <fstream>

#include "array2d.h"
#include "array4d.h"
#include "ptr_util.h"


namespace xla
{
   typedef unsigned char UChar8;

   struct BITMAPFILEHEADER
   {
      BITMAPFILEHEADER();

      unsigned short bfType;
      unsigned int   bfSize;
      unsigned short bfReserved1;
      unsigned short bfReserved2;
      unsigned int   bfOffBits;
   }; // 14 bytes

   inline
   BITMAPFILEHEADER::BITMAPFILEHEADER()
      : bfType(0x4d42)
      , bfSize(0)
      , bfReserved1(0)
      , bfReserved2(0)
      , bfOffBits(54)
   {
   }



   struct BITMAPINFOHEADER
   {
      BITMAPINFOHEADER();

      unsigned int    biSize;
      int             biWidth;
      int             biHeight;
      unsigned short  biPlanes;
      unsigned short  biBitCount;
      unsigned int    biCompression;
      unsigned int    biSizeImage;
      int             biXPelsPerMeter;
      int             biYPelsPerMeter;
      unsigned int    biClrUsed;
      unsigned int    biClrImportant;
   }; // 40 bytes

   inline
   BITMAPINFOHEADER::BITMAPINFOHEADER()
      : biSize(40)
      , biWidth(0)
      , biHeight(0)
      , biPlanes(1)
      , biBitCount(24)
      , biCompression(0)
      , biSizeImage(0)
      , biXPelsPerMeter(2835)
      , biYPelsPerMeter(2835)
      , biClrUsed(0)
      , biClrImportant(0)
   {
   }

   class image_load_error //: public dlib::error 
   { 
   public: 
      image_load_error(const std::string& str) 
      //: error(EIMAGE_LOAD, str)
      {}
   };

   struct rgb_pixel
   {
      UChar8 red;
      UChar8 green;
      UChar8 blue;
   };

   inline
   unsigned short read_u16(std::streambuf& in)
   {
      UChar8 b0, b1;

      in.sgetn(reinterpret_cast<char*>(&b0), 1);
      in.sgetn(reinterpret_cast<char*>(&b1), 1);

      return ((b1 << 8) | b0);
   }


   inline
   unsigned int read_u32(std::streambuf& in)
   {
      UChar8 b0, b1, b2, b3;

      in.sgetn(reinterpret_cast<char*>(&b0), 1);
      in.sgetn(reinterpret_cast<char*>(&b1), 1);
      in.sgetn(reinterpret_cast<char*>(&b2), 1);
      in.sgetn(reinterpret_cast<char*>(&b3), 1);

      return ((((((b3 << 8) | b2) << 8) | b1) << 8) | b0);
   }

   std::unique_ptr<xla::Array4D <UChar8> > load_bmp(const std::string& file_name);

   inline
   void write_header(const BITMAPFILEHEADER& header, std::streambuf& out)
   {  // write 14 bytes
      out.sputn(reinterpret_cast<const char*>(&header.bfType), 2);
      out.sputn(reinterpret_cast<const char*>(&header.bfSize), 4);
      out.sputn(reinterpret_cast<const char*>(&header.bfReserved1), 2);
      out.sputn(reinterpret_cast<const char*>(&header.bfReserved2), 2);
      out.sputn(reinterpret_cast<const char*>(&header.bfOffBits), 4);
   }

   inline
   void write_info(const BITMAPINFOHEADER& info, std::streambuf& out)
   {  // write 40 bytes
      out.sputn(reinterpret_cast<const char*>(&info.biSize),         4);
      out.sputn(reinterpret_cast<const char*>(&info.biWidth),        4);
      out.sputn(reinterpret_cast<const char*>(&info.biHeight),       4);
      out.sputn(reinterpret_cast<const char*>(&info.biPlanes),       2);
      out.sputn(reinterpret_cast<const char*>(&info.biBitCount),     2);
      out.sputn(reinterpret_cast<const char*>(&info.biCompression),  4);
      out.sputn(reinterpret_cast<const char*>(&info.biSizeImage),    4);
      out.sputn(reinterpret_cast<const char*>(&info.biXPelsPerMeter), 4);
      out.sputn(reinterpret_cast<const char*>(&info.biYPelsPerMeter), 4);
      out.sputn(reinterpret_cast<const char*>(&info.biClrUsed),      4);
      out.sputn(reinterpret_cast<const char*>(&info.biClrImportant), 4);
   }
}

#endif // IMAGE_LOADER_BMP

