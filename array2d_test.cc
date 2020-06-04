/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "array2d.h"

//#include "tensorflow/compiler/xla/array2d.h"

#include <initializer_list>

//#include "test.h"
#include "test_helpers.h"

#include "reference_util.h"

//#include "tensorflow/compiler/xla/test.h"

namespace xla {
namespace {


		void DefaultCtor() 
		{
		   Array2D<int> empty;
		   EXPECT_EQ(empty.n1(), 0);
		   EXPECT_EQ(empty.n2(), 0);
		   EXPECT_EQ(empty.num_elements(), 0);
		}

		void UninitializedDimsCtor() 
		{
		   Array2D<int> uninit(2, 3);
		   EXPECT_EQ(uninit.n1(), 2);
		   EXPECT_EQ(uninit.n2(), 3);
		   EXPECT_EQ(uninit.num_elements(), 6);
		}

		void FillCtor() 
		{
		   Array2D<int> fullof7(2, 3, 7);

		   EXPECT_EQ(fullof7.n1(), 2);
		   EXPECT_EQ(fullof7.n2(), 3);

		   for (int64 n1 = 0; n1 < fullof7.n1(); ++n1) 
         {
		      for (int64 n2 = 0; n2 < fullof7.n2(); ++n2) 
            {
		         EXPECT_EQ(fullof7(n1, n2), 7);
		      }
		   }
		}

		void InitializerListCtor() 
		{
		   Array2D<int> arr = {{1, 2, 3}, {4, 5, 6}};

		   EXPECT_EQ(arr.n1(), 2);
		   EXPECT_EQ(arr.n2(), 3);

		   EXPECT_EQ(arr(0, 0), 1);
		   EXPECT_EQ(arr(0, 1), 2);
		   EXPECT_EQ(arr(0, 2), 3);
		   EXPECT_EQ(arr(1, 0), 4);
		   EXPECT_EQ(arr(1, 1), 5);
		   EXPECT_EQ(arr(1, 2), 6);
		}

		void Accessors() 
		{
		   Array2D<int> arr = {{1, 2, 3}, {4, 5, 6}};

		   EXPECT_EQ(arr.n1(), 2);
		   EXPECT_EQ(arr.n2(), 3);
		   EXPECT_EQ(arr.height(), 2);
		   EXPECT_EQ(arr.width(), 3);
		   EXPECT_EQ(arr.num_elements(), 6);
		}

		void IndexingReadWrite() 
		{
		   Array2D<int> arr = {{1, 2, 3}, {4, 5, 6}};

		   EXPECT_EQ(arr(1, 1), 5);
		   EXPECT_EQ(arr(1, 2), 6);
		   arr(1, 1) = 51;
		   arr(1, 2) = 61;
		   EXPECT_EQ(arr(1, 1), 51);
		   EXPECT_EQ(arr(1, 2), 61);
		}

		void IndexingReadWriteBool() 
		{
         //  TODO:
         /*
		   Array2D<bool> arr = {{false, true, false}, {true, true, false}};

		   EXPECT_EQ(arr(1, 1), true);
		   EXPECT_EQ(arr(1, 2), false);
		   arr(1, 1) = false;
		   arr(1, 2) = true;
		   EXPECT_EQ(arr(1, 1), false);
		   EXPECT_EQ(arr(1, 2), true);
         */
		}

		void Fill() 
		{
		   Array2D<int> fullof7(2, 3, 7);
		   for (int64 n1 = 0; n1 < fullof7.n1(); ++n1) 
         {
		      for (int64 n2 = 0; n2 < fullof7.n2(); ++n2) 
            {
		         EXPECT_EQ(fullof7(n1, n2), 7);
		      }
		   }

		   fullof7.Fill(11);
		   for (int64 n1 = 0; n1 < fullof7.n1(); ++n1) 
         {
		      for (int64 n2 = 0; n2 < fullof7.n2(); ++n2) 
            {
		         EXPECT_EQ(fullof7(n1, n2), 11);
		      }
		   }
		}

		void DataPointer()
		{
		   Array2D<int> arr = {{1, 2, 3}, {4, 5, 6}};

		   EXPECT_EQ(arr.data()[0], 1);
		}

		void Linspace() 
		{
		   auto arr = MakeLinspaceArray2D(1.0, 3.5, 3, 2);

		   EXPECT_EQ(arr->n1(), 3);
		   EXPECT_EQ(arr->n2(), 2);

		   EXPECT_FLOAT_EQ((*arr)(0, 0), 1.0);
		   EXPECT_FLOAT_EQ((*arr)(0, 1), 1.5);
		   EXPECT_FLOAT_EQ((*arr)(1, 0), 2.0);
		   EXPECT_FLOAT_EQ((*arr)(1, 1), 2.5);
		   EXPECT_FLOAT_EQ((*arr)(2, 0), 3.0);
		   EXPECT_FLOAT_EQ((*arr)(2, 1), 3.5);
		}

		void Stringification() 
		{
		   auto arr = MakeLinspaceArray2D(1.0, 3.5, 3, 2);
		   const string expected = R"([[1, 1.5],
		   [2, 2.5],
		   [3, 3.5]])";
		   EXPECT_EQ(expected, arr->ToString());
		}


void testMatMul1()
{
   auto lhs = xla::MakeUnique<xla::Array2D<float>>(2, 3);
   auto rhs = xla::MakeUnique<xla::Array2D<float>>(3, 4);
   auto result_success = xla::MakeUnique<xla::Array2D<float>>(lhs->height(), rhs->width());

   (*lhs)(0, 0) = 1.f;
   (*lhs)(0, 1) = 3.f;
   (*lhs)(0, 2) = 2.f;
   (*lhs)(1, 0) = 0.f;
   (*lhs)(1, 1) = 4.f;
   (*lhs)(1, 2) = -1.f;

   (*rhs)(0, 0) = 2.f;
   (*rhs)(0, 1) = 0.f;
   (*rhs)(0, 2) = -1.f;
   (*rhs)(0, 3) = 1.f;
   (*rhs)(1, 0) = 3.f;
   (*rhs)(1, 1) = -2.f;
   (*rhs)(1, 2) = 1.f;
   (*rhs)(1, 3) = 2.f;
   (*rhs)(2, 0) = 0.f;
   (*rhs)(2, 1) = 1.f;
   (*rhs)(2, 2) = 2.f;
   (*rhs)(2, 3) = 3.f;

   (*result_success)(0, 0) = 11.f;
   (*result_success)(0, 1) = -4.f;
   (*result_success)(0, 2) = 6.f;
   (*result_success)(0, 3) = 13.f;
   (*result_success)(1, 0) = 12.f;
   (*result_success)(1, 1) = -9.f;
   (*result_success)(1, 2) = 2.f;
   (*result_success)(1, 3) = 5.f;

   //auto result = xla::MakeUnique<xla::Array2D<float>>(lhs->height(), rhs->width());
   //xla::MatrixMul<float>(*lhs, *rhs, *result);

   auto result = xla::MakeMatrixMul<float>(*lhs, *rhs);

   //////////////////////////////////////////////////////////////////////////

   ASSERT_TRUE(result->width() == result_success->width()) << "= width";
   assert(result->height() == result_success->height());

   assert(lhs->width() == rhs->height());

   for (xla::int64 i = 0; i < lhs->height(); i++)
   {
      for (xla::int64 j = 0; j < rhs->width(); j++)
      {
         const float rr = (*result)(i, j);
         assert((*result)(i, j) == (*result_success)(i, j));
      }
   }
}

void testMatMul2()
{
   auto lhs = xla::MakeUnique<xla::Array2D<float>>(3, 2);
   auto rhs = xla::MakeUnique<xla::Array2D<float>>(2, 4);
   auto result_success = xla::MakeUnique<xla::Array2D<float>>(lhs->height(), rhs->width());

   (*lhs)(0, 0) = -1.f;
   (*lhs)(0, 1) = 3.f;
   (*lhs)(1, 0) = 0.f;
   (*lhs)(1, 1) = 1.f;
   (*lhs)(2, 0) = 2.f;
   (*lhs)(2, 1) = -2.f;

   (*rhs)(0, 0) = 0.f;
   (*rhs)(0, 1) = 2.f;
   (*rhs)(0, 2) = 0.f;
   (*rhs)(0, 3) = -1.f;
   (*rhs)(1, 0) = 1.f;
   (*rhs)(1, 1) = -3.f;
   (*rhs)(1, 2) = 4.f;
   (*rhs)(1, 3) = 0.f;

   (*result_success)(0, 0) = 3.f;
   (*result_success)(0, 1) = -11.f;
   (*result_success)(0, 2) = 12.f;
   (*result_success)(0, 3) = 1.f;
   (*result_success)(1, 0) = 1.f;
   (*result_success)(1, 1) = -3.f;
   (*result_success)(1, 2) = 4.f;
   (*result_success)(1, 3) = 0.f;
   (*result_success)(2, 0) = -2.f;
   (*result_success)(2, 1) = 10.f;
   (*result_success)(2, 2) = -8.f;
   (*result_success)(2, 3) = -2.f;

   //auto result = xla::MakeUnique<xla::Array2D<float>>(lhs->height(), rhs->width());
   //xla::MatrixMul<float>(*lhs, *rhs, *result);

   auto result = xla::MakeMatrixMul<float>(*lhs, *rhs);

   //////////////////////////////////////////////////////////////////////////

   assert(result->width() == result_success->width());
   assert(result->height() == result_success->height());

   assert(lhs->width() == rhs->height());

   for (xla::int64 i = 0; i < lhs->height(); i++)
   {
      for (xla::int64 j = 0; j < rhs->width(); j++)
      {
         const float rr = (*result)(i, j);
         assert((*result)(i, j) == (*result_success)(i, j));
      }
   }
}

void testMatMul3()
{
   auto lhs = xla::MakeUnique<xla::Array2D<float>>(3, 1);
   auto rhs = xla::MakeUnique<xla::Array2D<float>>(1, 3);
   auto result_success = xla::MakeUnique<xla::Array2D<float>>(lhs->height(), rhs->width());

   (*lhs)(0, 0) = 2.f;
   (*lhs)(1, 0) = 0.f;
   (*lhs)(2, 0) = 1.f;

   (*rhs)(0, 0) = -3.f;
   (*rhs)(0, 1) = 4.f;
   (*rhs)(0, 2) = 5.f;

   (*result_success)(0, 0) = -6.f;
   (*result_success)(0, 1) = 8.f;
   (*result_success)(0, 2) = 10.f;
   (*result_success)(1, 0) = 0.f;
   (*result_success)(1, 1) = 0.f;
   (*result_success)(1, 2) = 0.f;
   (*result_success)(2, 0) = -3.f;
   (*result_success)(2, 1) = 4.f;
   (*result_success)(2, 2) = 5.f;

   //auto result = xla::MakeUnique<xla::Array2D<float>>(lhs->height(), rhs->width());
   //xla::MatrixMul<float>(*lhs, *rhs, *result);

   auto result = xla::MakeMatrixMul<float>(*lhs, *rhs);

   //////////////////////////////////////////////////////////////////////////

   assert(result->width() == result_success->width());
   assert(result->height() == result_success->height());

   assert(lhs->width() == rhs->height());

   for (xla::int64 i = 0; i < lhs->height(); i++)
   {
      for (xla::int64 j = 0; j < rhs->width(); j++)
      {
         const float rr = (*result)(i, j);
         assert((*result)(i, j) == (*result_success)(i, j));
      }
   }
}

void testMatMul4()
{
   auto lhs = xla::MakeUnique<xla::Array2D<float>>(1, 3);
   auto rhs = xla::MakeUnique<xla::Array2D<float>>(3, 1);
   auto result_success = xla::MakeUnique<xla::Array2D<float>>(lhs->height(), rhs->width());

   (*lhs)(0, 0) = 2.f;
   (*lhs)(0, 1) = 0.f;
   (*lhs)(0, 2) = 1.f;

   (*rhs)(0, 0) = -3.f;
   (*rhs)(1, 0) = 4.f;
   (*rhs)(2, 0) = 5.f;

   (*result_success)(0, 0) = -1.f;

   //auto result = xla::MakeUnique<xla::Array2D<float>>(lhs->height(), rhs->width());
   //xla::MatrixMul2D<float>(*lhs, *rhs, *result);

   auto result = xla::MakeMatrixMul<float>(*lhs, *rhs);

   //////////////////////////////////////////////////////////////////////////

   assert(result->width() == result_success->width());
   assert(result->height() == result_success->height());

   assert(lhs->width() == rhs->height());

   for (xla::int64 i = 0; i < lhs->height(); i++)
   {
      for (xla::int64 j = 0; j < rhs->width(); j++)
      {
         const float rr = (*result)(i, j);
         assert((*result)(i, j) == (*result_success)(i, j));
      }
   }
}


void testMatMul_3D()
{
   auto lhs = xla::MakeUnique<xla::Array3D<float>>(1, 3, 2);
   auto rhs = xla::MakeUnique<xla::Array3D<float>>(1, 2, 4);
   auto result_success = xla::MakeUnique<xla::Array3D<float>>(1, lhs->Height(), rhs->Width());

   (*lhs)(0, 0, 0) = -1.f;
   (*lhs)(0, 0, 1) =  3.f;
   (*lhs)(0, 1, 0) =  0.f;
   (*lhs)(0, 1, 1) =  1.f;
   (*lhs)(0, 2, 0) =  2.f;
   (*lhs)(0, 2, 1) = -2.f;

   (*rhs)(0, 0, 0) =  0.f;
   (*rhs)(0, 0, 1) =  2.f;
   (*rhs)(0, 0, 2) =  0.f;
   (*rhs)(0, 0, 3) = -1.f;
   (*rhs)(0, 1, 0) =  1.f;
   (*rhs)(0, 1, 1) = -3.f;
   (*rhs)(0, 1, 2) =  4.f;
   (*rhs)(0, 1, 3) =  0.f;

   (*result_success)(0, 0, 0) =  3.f;
   (*result_success)(0, 0, 1) = -11.f;
   (*result_success)(0, 0, 2) =  12.f;
   (*result_success)(0, 0, 3) =  1.f;
   (*result_success)(0, 1, 0) =  1.f;
   (*result_success)(0, 1, 1) = -3.f;
   (*result_success)(0, 1, 2) =  4.f;
   (*result_success)(0, 1, 3) =  0.f;
   (*result_success)(0, 2, 0) = -2.f;
   (*result_success)(0, 2, 1) =  10.f;
   (*result_success)(0, 2, 2) = -8.f;
   (*result_success)(0, 2, 3) = -2.f;

   //auto result = xla::MakeUnique<xla::Array2D<float>>(lhs->height(), rhs->width());
   //xla::MatrixMul<float>(*lhs, *rhs, *result);

   auto result = xla::MakeMatrixMul<float>(*lhs, *rhs);

   //////////////////////////////////////////////////////////////////////////

   assert(result->Width() == result_success->Width());
   assert(result->Height() == result_success->Height());

   assert(lhs->Width() == rhs->Height());

   for (xla::int64 i = 0; i < lhs->Height(); i++)
   {
      for (xla::int64 j = 0; j < rhs->Width(); j++)
      {
         const float rr = (*result)(0, i, j);
         assert((*result)(0, i, j) == (*result_success)(0, i, j));
      }
   }
}

/*
template <typename T>
xla::Array2D<T> MatrixCrossMul(const xla::Array2D<T>& lhs, const xla::Array2D<T>& rhs)
{
   xla::Array2D<T> matrix_out(lhs.n1()*rhs.n1(), lhs.n2()*rhs.n2());

   for (int i = 0; i < lhs.n1(); ++i)  // cols
   {
      for (int j = 0; j < lhs.n2(); ++j)  // rows
      {
         for (int k = 0; k < rhs.n1(); ++k)  // cols
         {
            for (int n = 0; n < rhs.n2(); ++n)  // rows
            {
               matrix_out(i*rhs.n1() + k, j*rhs.n2() + n) = lhs(i, j) * rhs(k, n);
            }
         }
      }
   }

   return matrix_out;
}
*/


}  // namespace
}  // namespace xla

