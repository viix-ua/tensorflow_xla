
#ifndef MAX_POOL_H_
#define MAX_POOL_H_

#include "array4d.h"
#include "padding.h"
#include "test_helpers.h"

namespace tensorflow
{

   const int kInvalidMaxPoolingIndex = -1;

   // pooling_ops_common.h

   // A helper class to manage sizes and shapes for pooling operations.
   struct PoolParameters
   {
      // Updates context->status if there is an invalid input.
      PoolParameters(const std::vector<int32>& ksize,
         const std::vector<int32>& stride, xla::Padding padding,
         const xla::Array4D<double>& tensor_in_shape);

      // Returns the shape of the output for "forward" pooling operations.
      xla::Array4D<double> forward_output_shape();

      int depth;

      int tensor_in_cols;
      int tensor_in_rows;
      int tensor_in_batch;

      int window_rows;
      int window_cols;
      int depth_window;

      int row_stride;
      int col_stride;
      int depth_stride;

      int64 out_height;
      int64 out_width;
      int out_depth;

      int64 pad_rows;
      int64 pad_cols;
      int pad_depth;

      //TensorFormat data_format;
   };

   // maxpooling_op.h
   // performed by:
   // core/kernels/eigen_pooling.h
   // core/kernels/eigen_pooling_test.cc
   namespace functor
   {
      template <typename T>
      struct SpatialMaxPooling
      {
         void operator()(
            xla::Array4D& output,
            const xla::Array4D& input,
            int window_rows, int window_cols,
            int row_stride, int col_stride,
            const xla::Padding& padding)
         {
            // Because we swap the layout, we swap the row/cols as well
            // output.swap_layout().device(d) =
            //   Eigen::SpatialMaxPooling(input.swap_layout(), window_cols, window_rows,
            //      col_stride, row_stride, padding);
         }
      };
   }  // namespace functor


   // core/kernels/pooling_ops_common.h

   // An implementation of MaxPooling (forward).
   template <typename T>
   class MaxPoolingOp   // : public OpKernel 
   {
   public:
      MaxPoolingOp()
      {
         // Default MaxPoolingOp only supports NHWC.
         //data_format_ = FORMAT_NHWC;
         ASSERT_TRUE(ksize_.size() == 4);

         ASSERT_TRUE(stride_.size() == 4);
      }

      void Compute() override
      {
         const Tensor& tensor_in = context->input(0);
         PoolParameters params{ ksize_, stride_, padding_, /*FORMAT_NHWC, */ tensor_in };
         if (!context->status().ok())
         {
            return;
         }

         Tensor* output = nullptr;
         //OP_REQUIRES_OK(context, context->allocate_output(
         //   0, params.forward_output_shape(), &output));

         if (params.depth_window > 1)
         {
            // Validate spec against the current implementation.  A
            // relaxation of these requirements would be ideal.
            //OP_REQUIRES(context, params.depth % params.depth_window == 0,
            //   errors::Unimplemented(
            //      "Depthwise max pooling requires "
            //      "the depth window to evenly divide the input depth."));
            //OP_REQUIRES(
            //   context, params.depth_window == params.depth_stride,
            //   errors::Unimplemented("Depthwise max pooling requires "
            //      "the depth window to equal the depth stride."));

            DepthwiseMaxPool(output, tensor_in, params);
         }
         else
         {
            SpatialMaxPool(output, tensor_in, params, padding_);
         }
      }

   private:
      // Single-threaded implementation of DepthwiseMaxPool which
      // does not handle all of the same options as SpatialMaxPool
      // (strict assumptions on no padding, stride).
      //
      // TODO(vrv): implement a more general depthwise-max pool that works
      // on GPU as well.
      void DepthwiseMaxPool(xla::Array4D<T>* output,
         const xla::Array4D<T>& tensor_in, const PoolParameters& params)
      {
         /*
         Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
            in_by_pool(tensor_in.flat<T>().data(), params.depth_window,
               tensor_in.NumElements() / params.depth_window);
         Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> out_by_pool(
            output->flat<T>().data(), 1, output->NumElements());
         out_by_pool = in_by_pool.colwise().maxCoeff();
         */
      }

      template <typename T>
      void SpatialMaxPool(
         const xla::Array4D<T>& tensor_in,
         const PoolParameters& params,
         const xla::Padding& padding)
      {
         // On GPU, use Eigen's Spatial Max Pooling.  On CPU, use an
         // EigenMatrix version that is currently faster than Eigen's
         // Spatial MaxPooling implementation.
         //
         // TODO(vrv): Remove this once we no longer need it.
         //if (std::is_same<Device, GPUDevice>::value) {
         //   Eigen::PaddingType pt = BrainPadding2EigenPadding(padding);
         //   functor::SpatialMaxPooling<Device, T>()(
         //      context->eigen_device<Device>(), output->tensor<T, 4>(),
         //      tensor_in.tensor<T, 4>(), params.window_rows, params.window_cols,
         //      params.row_stride, params.col_stride, pt);
         //}
         //else 
         {
            //typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
            //   ConstEigenMatrixMap;
            //typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
            //   EigenMatrixMap;

            //ConstEigenMatrixMap in_mat(tensor_in.flat<T>().data(), params.depth,
            //   params.tensor_in_cols * params.tensor_in_rows *
            //   params.tensor_in_batch);
            //EigenMatrixMap out_mat(
            //   output->flat<T>().data(), params.depth,
            //   params.out_width * params.out_height * params.tensor_in_batch);


            // The following code basically does the following:
            // 1. Flattens the input and output tensors into two dimensional arrays.
            //    tensor_in_as_matrix:
            //      depth by (tensor_in_cols * tensor_in_rows * tensor_in_batch)
            //    output_as_matrix:
            //      depth by (out_width * out_height * tensor_in_batch)
            //
            // 2. Walks through the set of columns in the flattened
            // tensor_in_as_matrix,
            //    and updates the corresponding column(s) in output_as_matrix with the
            //    max value.
            auto shard = [&params, &in_mat, &out_mat](int64 start, int64 limit)
            {

               const int32 in_rows = params.tensor_in_rows;
               const int32 in_cols = params.tensor_in_cols;
               const int32 pad_rows = params.pad_rows;
               const int32 pad_cols = params.pad_cols;
               const int32 window_rows = params.window_rows;
               const int32 window_cols = params.window_cols;
               const int32 row_stride = params.row_stride;
               const int32 col_stride = params.col_stride;
               const int32 out_height = params.out_height;
               const int32 out_width = params.out_width;

               {
                  // Initializes the output tensor with MIN<T>.
                  const int32 output_image_size = out_height * out_width * params.depth;
                  //EigenMatrixMap out_shard(out_mat.data() + start * output_image_size,
                  //   1, (limit - start) * output_image_size);
                  //out_shard.setConstant(Eigen::NumTraits<T>::lowest());
               }

               for (int32 b = start; b < limit; ++b)
               {
                  const int32 out_offset_batch = b * out_height;
                  for (int32 h = 0; h < in_rows; ++h)
                  {
                     for (int32 w = 0; w < in_cols; ++w)
                     {
                        // (h_start, h_end) * (w_start, w_end) is the range that the input
                        // vector projects to.
                        const int32 hpad = h + pad_rows;
                        const int32 wpad = w + pad_cols;
                        const int32 h_start = (hpad < window_rows)
                           ? 0
                           : (hpad - window_rows) / row_stride + 1;
                        const int32 h_end = std::min(hpad / row_stride + 1, out_height);
                        const int32 w_start = (wpad < window_cols)
                           ? 0
                           : (wpad - window_cols) / col_stride + 1;
                        const int32 w_end = std::min(wpad / col_stride + 1, out_width);
                        // compute elementwise max
                        const int32 in_offset = (b * in_rows + h) * in_cols + w;
                        for (int32 ph = h_start; ph < h_end; ++ph)
                        {
                           const int32 out_offset_base =
                              (out_offset_batch + ph) * out_width;
                           for (int32 pw = w_start; pw < w_end; ++pw)
                           {
                              const int32 out_offset = out_offset_base + pw;
                              /*
                              out_mat.col(out_offset) =
                                 out_mat.col(out_offset).cwiseMax(in_mat.col(in_offset));
                                 */
                           }
                        }
                     }
                  }
               }
            };

            // TODO(andydavis) Consider sharding across batch x rows x cols.
            // TODO(andydavis) Consider a higher resolution shard cost model.
            const int64 shard_cost =
               params.tensor_in_rows * params.tensor_in_cols * params.depth;
            //Shard(worker_threads.num_threads, worker_threads.workers,
            //   params.tensor_in_batch, shard_cost, shard);
         }
      }

      std::vector<int32> ksize_;
      std::vector<int32> stride_;
      xla::Padding padding_;
      //TensorFormat data_format_;
   };

   /*
   template <typename Device, typename T>
   static void SpatialMaxPoolWithArgMaxHelper(
      OpKernelContext* context, Tensor* output, Tensor* output_arg_max,
      Tensor* input_backprop, const Tensor& tensor_in, const Tensor& out_backprop,
      const PoolParameters& params, const Padding& padding)
   {
      //typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      //   ConstEigenMatrixMap;
      //typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
      //   EigenMatrixMap;
      //typedef Eigen::Map<Eigen::Matrix<int64, Eigen::Dynamic, Eigen::Dynamic>>
      //   EigenIndexMatrixMap;

      //ConstEigenMatrixMap in_mat(
      //   tensor_in.flat<T>().data(), params.depth,
      //   params.tensor_in_cols * params.tensor_in_rows * params.tensor_in_batch);
      //EigenMatrixMap out_mat(
      //   output->flat<T>().data(), params.depth,
      //   params.out_width * params.out_height * params.tensor_in_batch);
      //EigenIndexMatrixMap out_arg_max_mat(
      //   output_arg_max->flat<int64>().data(), params.depth,
      //   params.out_width * params.out_height * params.tensor_in_batch);

      //const DeviceBase::CpuWorkerThreads& worker_threads =
      //   *(context->device()->tensorflow_cpu_worker_threads());
      ///////////////////////////////////////////////////////////////////////////////////////

      // The following code basically does the following:
      // 1. Flattens the input and output tensors into two dimensional arrays.
      //    tensor_in_as_matrix:
      //      depth by (tensor_in_cols * tensor_in_rows * tensor_in_batch)
      //    output_as_matrix:
      //      depth by (out_width * out_height * tensor_in_batch)
      //
      // 2. Walks through the set of columns in the flattened tensor_in_as_matrix,
      //    and updates the corresponding column(s) in output_as_matrix with the
      //    max value.
      auto shard = [&params, &in_mat, &out_mat, &out_arg_max_mat, &input_backprop,
         &output_arg_max, &out_backprop](int64 start, int64 limit)
      {

         const int32 depth = params.depth;
         const int32 in_rows = params.tensor_in_rows;
         const int32 in_cols = params.tensor_in_cols;
         const int32 pad_rows = params.pad_rows;
         const int32 pad_cols = params.pad_cols;
         const int32 window_rows = params.window_rows;
         const int32 window_cols = params.window_cols;
         const int32 row_stride = params.row_stride;
         const int32 col_stride = params.col_stride;
         const int32 out_height = params.out_height;
         const int32 out_width = params.out_width;

         {
            // Initializes the output tensor with MIN<T>.
            const int32 output_image_size = out_height * out_width * depth;
            EigenMatrixMap out_shard(out_mat.data() + start * output_image_size, 1,
               (limit - start) * output_image_size);
            out_shard.setConstant(Eigen::NumTraits<T>::lowest());
            EigenIndexMatrixMap out_arg_max_shard(
               out_arg_max_mat.data() + start * output_image_size, 1,
               (limit - start) * output_image_size);
            out_arg_max_shard.setConstant(kInvalidMaxPoolingIndex);
         }

         for (int64 b = start; b < limit; ++b)
         {
            for (int h = 0; h < in_rows; ++h)
            {
               for (int w = 0; w < in_cols; ++w)
               {
                  // (h_start, h_end) * (w_start, w_end) is the range that the input
                  // vector projects to.
                  const int hpad = h + pad_rows;
                  const int wpad = w + pad_cols;
                  const int h_start =
                     (hpad < window_rows) ? 0 : (hpad - window_rows) / row_stride + 1;
                  const int h_end = std::min(hpad / row_stride + 1, out_height);
                  const int w_start =
                     (wpad < window_cols) ? 0 : (wpad - window_cols) / col_stride + 1;
                  const int w_end = std::min(wpad / col_stride + 1, out_width);
                  // compute elementwise max
                  const int64 in_index = (b * in_rows + h) * in_cols + w;
                  for (int ph = h_start; ph < h_end; ++ph)
                  {
                     const int64 out_index_base = (b * out_height + ph) * out_width;
                     for (int pw = w_start; pw < w_end; ++pw)
                     {
                        const int64 out_index = out_index_base + pw;
                        /// NOTES(zhengxq): not using the eigen matrix operation for
                        /// now.
                        for (int d = 0; d < depth; ++d)
                        {
                           const T& input_ref = in_mat.coeffRef(d, in_index);
                           T& output_ref = out_mat.coeffRef(d, out_index);
                           int64& out_arg_max_ref = out_arg_max_mat.coeffRef(d, out_index);
                           if (output_ref < input_ref ||
                              out_arg_max_ref == kInvalidMaxPoolingIndex)
                           {
                              output_ref = input_ref;
                              int64 input_offset = in_index * depth + d;
                              out_arg_max_ref = input_offset;
                           }
                        }
                     }
                  }
               }
            }
         }

         {
            auto input_backprop_flat = input_backprop->flat<T>();
            auto out_arg_max_flat = output_arg_max->flat<int64>();
            auto out_backprop_flat = out_backprop.flat<T>();

            // Initialize output to 0.
            const int64 in_size = in_rows * in_cols * depth;
            const int64 in_start = start * in_size;
            const int64 in_end = limit * in_size;
            EigenMatrixMap in_shard(input_backprop_flat.data() + in_start, 1,
               in_end - in_start);
            in_shard.setConstant(T(0));

            // Backpropagate.
            const int out_size = out_height * out_width * depth;
            const int out_start = start * out_size;
            const int out_end = limit * out_size;
            for (int index = out_start; index < out_end; ++index)
            {
               int input_backprop_index = out_arg_max_flat(index);
               // Although this check is in the inner loop, it is worth its value
               // so we don't end up with memory corruptions. Our benchmark shows that
               // the performance impact is quite small
               CHECK(input_backprop_index >= in_start && input_backprop_index < in_end)
                  << "Invalid input backprop index: " << input_backprop_index << ", "
                  << in_start << ", " << in_end;
               input_backprop_flat(input_backprop_index) += out_backprop_flat(index);
            }
         }

      };

      const int64 shard_cost = params.tensor_in_rows * params.tensor_in_cols *
         params.depth * params.window_rows *
         params.window_cols;
      //Shard(worker_threads.num_threads, worker_threads.workers,
      //   params.tensor_in_batch, shard_cost, shard);
   }
   */

}  // tensorflow

#endif

