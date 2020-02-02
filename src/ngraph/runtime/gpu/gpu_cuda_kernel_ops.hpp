//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include "ngraph/op/abs.hpp"
#include "ngraph/op/acos.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/all.hpp"
#include "ngraph/op/allreduce.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/any.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/op/asin.hpp"
#include "ngraph/op/atan2.hpp"
#include "ngraph/op/atan.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/binary_convolution.hpp"
#include "ngraph/op/broadcast_distributed.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/ceiling.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/convert_like.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/cosh.hpp"
#include "ngraph/op/cos.hpp"
#include "ngraph/op/crop_and_resize.hpp"
#include "ngraph/op/cum_sum.hpp"
#include "ngraph/op/deformable_convolution.hpp"
#include "ngraph/op/deformable_psroi_pooling.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/embedding_lookup.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/erf.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/floor_mod.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/gather_nd.hpp"
#include "ngraph/op/gather_tree.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/greater_eq.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/non_max_suppression.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/passthrough.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/quantized_convolution.hpp"
#include "ngraph/op/quantized_dot.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/recv.hpp"
#include "ngraph/op/reduce_logical_and.hpp"
#include "ngraph/op/reduce_logical_or.hpp"
#include "ngraph/op/reduce_mean.hpp"
#include "ngraph/op/reduce_prod.hpp"
#include "ngraph/op/reduce_sum.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/round.hpp"
#include "ngraph/op/scatter_add.hpp"
#include "ngraph/op/scatter_nd_add.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/send.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/sinh.hpp"
#include "ngraph/op/sin.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/stop_gradient.hpp"
#include "ngraph/op/strided_slice.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/op/tan.hpp"
#include "ngraph/op/tensor_iterator.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/op/variadic_split.hpp"
#include "ngraph/op/xor.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            enum class OpName
            {
                add,
                multiply,
                minimum,
                maximum
            };

            template <typename T>
            struct CudaOpMap;

            template <>
            struct CudaOpMap<ngraph::op::Abs>
            {
                static constexpr const char* op = "fabsf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Acos>
            {
                static constexpr const char* op = "acosf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Asin>
            {
                static constexpr const char* op = "asinf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Atan>
            {
                static constexpr const char* op = "atanf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Ceiling>
            {
                static constexpr const char* op = "ceilf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Cos>
            {
                static constexpr const char* op = "cosf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Cosh>
            {
                static constexpr const char* op = "coshf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Exp>
            {
                static constexpr const char* op = "expf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Floor>
            {
                static constexpr const char* op = "floorf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Log>
            {
                static constexpr const char* op = "logf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Max>
            {
                static constexpr const char* op = "fmaxf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Min>
            {
                static constexpr const char* op = "fminf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Sin>
            {
                static constexpr const char* op = "sinf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Sinh>
            {
                static constexpr const char* op = "sinhf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Sqrt>
            {
                static constexpr const char* op = "sqrtf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Tan>
            {
                static constexpr const char* op = "tanf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Tanh>
            {
                static constexpr const char* op = "tanhf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Power>
            {
                static constexpr const char* op = "powf";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Subtract>
            {
                static constexpr const char* op = "subtractf";
                static constexpr const char* math_kernel = "x0-x1";
                static constexpr const char* atomic = "atomicSub";
            };

            template <>
            struct CudaOpMap<ngraph::op::Divide>
            {
                static constexpr const char* op = "fdividef";
                static constexpr const char* math_kernel = nullptr;
            };

            template <>
            struct CudaOpMap<ngraph::op::Sign>
            {
                static constexpr const char* op = "sign";
                static constexpr const char* math_kernel = "(x0 > 0) - (x0 < 0)";
            };

            template <>
            struct CudaOpMap<ngraph::op::Convert>
            {
                static constexpr const char* op = "convert";
                static constexpr const char* math_kernel = "x0";
            };

            template <>
            struct CudaOpMap<ngraph::op::Equal>
            {
                static constexpr const char* op = "equal";
                static constexpr const char* math_kernel = "x0 == x1";
            };

            template <>
            struct CudaOpMap<ngraph::op::NotEqual>
            {
                static constexpr const char* op = "not_equal";
                static constexpr const char* math_kernel = "x0 != x1";
            };

            template <>
            struct CudaOpMap<ngraph::op::Greater>
            {
                static constexpr const char* op = "greater";
                static constexpr const char* math_kernel = "x0 > x1";
            };

            template <>
            struct CudaOpMap<ngraph::op::GreaterEq>
            {
                static constexpr const char* op = "greater_equal";
                static constexpr const char* math_kernel = "x0 >= x1";
            };

            template <>
            struct CudaOpMap<ngraph::op::Less>
            {
                static constexpr const char* op = "less";
                static constexpr const char* math_kernel = "x0 < x1";
            };

            template <>
            struct CudaOpMap<ngraph::op::LessEq>
            {
                static constexpr const char* op = "less_equal";
                static constexpr const char* math_kernel = "x0 <= x1";
            };

            template <>
            struct CudaOpMap<ngraph::op::Relu>
            {
                static constexpr const char* op = "relu";
                static constexpr const char* math_kernel = "fmaxf(0,x0)";
            };

            template <>
            struct CudaOpMap<ngraph::op::Not>
            {
                static constexpr const char* op = "logical_not";
                static constexpr const char* math_kernel = "!x0";
            };

            template <>
            struct CudaOpMap<ngraph::op::Negative>
            {
                static constexpr const char* op = "negative";
                static constexpr const char* math_kernel = "-x0";
            };

            template <>
            struct CudaOpMap<ngraph::op::Select>
            {
                static constexpr const char* op = "select";
                static constexpr const char* math_kernel = "(x0 == 0) ? x2 : x1";
            };

            template <>
            struct CudaOpMap<ngraph::op::ReluBackprop>
            {
                static constexpr const char* op = "relu_backprop";
                static constexpr const char* math_kernel = "x1 * int(x0 > 0)";
            };

            template <>
            struct CudaOpMap<ngraph::op::And>
            {
                static constexpr const char* op = "logical_and";
                static constexpr const char* math_kernel = "x0 & x1";
                static constexpr const char* atomic = "atomicAnd";
            };

            template <>
            struct CudaOpMap<ngraph::op::Or>
            {
                static constexpr const char* op = "logical_or";
                static constexpr const char* math_kernel = "x0 | x1";
                static constexpr const char* atomic = "atomicOr";
            };

            template <>
            struct CudaOpMap<ngraph::op::Add>
            {
                static constexpr const char* op = "add";
                static constexpr const char* math_kernel = "x0 + x1";
                static constexpr const char* atomic = "atomicAdd";
            };

            template <>
            struct CudaOpMap<ngraph::op::Multiply>
            {
                static constexpr const char* op = "mul";
                static constexpr const char* math_kernel = "x0 * x1";
            };

            template <>
            struct CudaOpMap<ngraph::op::Minimum>
            {
                static constexpr const char* op = "min";
                static constexpr const char* math_kernel = "x0 > x1 ? x1 : x0";
                static constexpr const char* atomic = "atomicMin";
            };

            template <>
            struct CudaOpMap<ngraph::op::Maximum>
            {
                static constexpr const char* op = "max";
                static constexpr const char* math_kernel = "x0 > x1 ? x0 : x1";
                static constexpr const char* atomic = "atomicMax";
            };

            template <>
            struct CudaOpMap<ngraph::op::Sigmoid>
            {
                static constexpr const char* op = "sigmoid";
                static constexpr const char* math_kernel = "1 / (1 + expf(-x0))";
            };

            template <>
            struct CudaOpMap<ngraph::op::SigmoidBackprop>
            {
                static constexpr const char* op = "sigmoid_backprop";
                static constexpr const char* math_kernel = "x1 / (2 + expf(-x0) + expf(x0))";
            };
        }
    }
}
