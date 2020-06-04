
// Linear Regression (LR)
// Stochastic Gradiend Descent (SGD) method

// https://github.com/therealAJ/lin-reg/blob/master/demo.ipynb

#ifndef TRAINER_BASE_LR_SGD_H
#define TRAINER_BASE_LR_SGD_H

#include <initializer_list>
#include <vector>

#include "array2d.h"

// linear regression with stochastic gradient descent trainer on y = f(x), for one variable

namespace xla
{
   template <typename TType>
   class TLinearGradientDescentTrainer
   {
      TType mResultSSE;

      // prediction, function y = dot(theta, x_test.row(id))
      static TType hypothesis(const std::vector<TType>& theta, const xla::Array2D<TType>& input, unsigned int rowId)
      {
         CHECK_EQ(input.width(), theta.size());

         TType h = 0.f;
         for (unsigned int i = 0; i < input.width(); i++)
         {
            h += input(rowId, i) * theta[i];
         }
         return h;
      }

      // reduce_sum
      static TType cost_function_sse(const xla::Array2D<TType>& X, const std::vector<TType>& y, const std::vector<TType>& theta)
      {
         TType sse = 0.f;

         CHECK_EQ(y.size(), X.height());

         const size_t m = y.size();

         for (size_t i = 0; i < m; i++)
         {
            TType h_i = hypothesis(theta, X, static_cast<unsigned int>(i));
            TType err = (h_i - y[i]);

            sse += (err*err);
         }
         return (0.5f * sse) / TType(m);
      }

      // calculate gradient of single component
      static TType gradient(const std::vector<TType>& theta, const xla::Array2D<TType>& X, const std::vector<TType>& y, unsigned int sub_i, unsigned int m)
      {
         TType sumErr = 0.f;

         for (unsigned int i = 0; i < m; i++)
         {
            TType h_i = hypothesis(theta, X, i);

            TType delta = (h_i - y[i]) * X(i, sub_i);
            sumErr += delta;
         }
         return (sumErr / TType(m));
      }


      static void step_gradient_descent(const xla::Array2D<TType>& X, const std::vector<TType>& y, std::vector<TType>& theta, unsigned int m, TType learn_rate)
      {
         std::vector<float> opt_theta = theta;

         const TType constant = learn_rate / TType(m);

         for (unsigned int i = 0; i < theta.size(); i++)
         {
            TType cost = TLinearGradientDescentTrainer::gradient(theta, X, y, i, m);

            TType updated_component = theta[i] - constant * cost;

            opt_theta[i] = updated_component;
         }

         // return updated theta
         theta = opt_theta;
      }


      static TType calculate(const xla::Array2D<TType>& X, const std::vector<TType>& y, std::vector<TType>& opt_theta, TType learn_rate, unsigned int iterations)
      {
         TType cost_function = TLinearGradientDescentTrainer::cost_function_sse(X, y, opt_theta);

         for (unsigned int i = 0; i < iterations; i++)
         {
            TLinearGradientDescentTrainer::step_gradient_descent(X, y, opt_theta, static_cast<unsigned int>(X.height()), learn_rate);

            cost_function = TLinearGradientDescentTrainer::cost_function_sse(X, y, opt_theta);
         }
         return cost_function;
      }

   public:


      TType getResultCost() const
      {
         return mResultSSE;
      }


      std::vector<TType> calculate(const std::vector<TType>& train_X, const std::vector<TType>& train_Y, const std::vector<TType>& theta, TType learn_rate, unsigned int iterations)
      {
         CHECK_EQ(train_Y.size(), train_X.size());

         xla::Array2D<TType> X(train_X.size(), 2);
         for (size_t n = 0; n < train_X.size(); n++)
         {
            X(n, 0) = TType(1.f);   // B
            X(n, 1) = train_X[n];   // K=1
         }

         std::vector<TType> opt_theta = theta;

         mResultSSE = calculate(X, train_Y, opt_theta, learn_rate, iterations);

         return opt_theta;
      }
   }; // TLinearGradientDescent

   void run()
   {
      //std::vector<float> train_X = { 3.f, 9.f, 2.f, 4.f, 8.f, 2.f, 12.f, 20.f };

      //std::vector<float> train_Y = { 12.f, 14.f, 10.f, 23.f, 20.f, 7.f, 13.f, 24.f };

      //std::vector<float> theta = { 0, 0 };

      //float learn_rate = 0.05f;
      //const unsigned int iterations = 10000;

      //xla::TLinearGradientDescentTrainer<float> gradient_descent;

      //std::vector<float> optimal_theta = gradient_descent.calculate(train_X, train_Y, theta, learn_rate, iterations);
   }

}  // ns


#endif
