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

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/quantized_convolution.hpp"
#include "quantization_utils.hpp"

namespace ngraph
{
    namespace builder
    {
        std::shared_ptr<Node>
            QuantizedConvolutionBuilder(const Output& input,
                                        const Output& filters,
                                        const Strides& window_movement_strides,
                                        const Strides& window_dilation_strides,
                                        const CoordinateDiff& padding_below,
                                        const CoordinateDiff& padding_above,
                                        const Strides& data_dilation_strides,
                                        const Output& min_input,
                                        const Output& max_input,
                                        const Output& min_filter,
                                        const Output& max_filter,
                                        const Output& min_output,
                                        const Output& max_output,
                                        const ngraph::element::Type& output_type,
                                        const ngraph::AxisSet& input_axes = ngraph::AxisSet{},
                                        const ngraph::AxisSet& filter_axes = ngraph::AxisSet{},
                                        const ngraph::AxisSet& output_axes = ngraph::AxisSet{});

        std::shared_ptr<Node>
            QuantizedConvolutionBiasBuilder(const Output& input,
                                            const Output& filters,
                                            const Output& bias,
                                            const Strides& window_movement_strides,
                                            const Strides& window_dilation_strides,
                                            const CoordinateDiff& padding_below,
                                            const CoordinateDiff& padding_above,
                                            const Strides& data_dilation_strides,
                                            const Output& min_input,
                                            const Output& max_input,
                                            const Output& min_filter,
                                            const Output& max_filter,
                                            const Output& min_output,
                                            const Output& max_output,
                                            const bool with_relu = false);

        std::shared_ptr<Node>
            QuantizedConvolutionReluBuilder(const Output& input,
                                            const Output& filters,
                                            const Strides& window_movement_strides,
                                            const Strides& window_dilation_strides,
                                            const CoordinateDiff& padding_below,
                                            const CoordinateDiff& padding_above,
                                            const Strides& data_dilation_strides,
                                            const Output& min_input,
                                            const Output& max_input,
                                            const Output& min_filter,
                                            const Output& max_filter,
                                            const Output& min_output,
                                            const Output& max_output);

        std::shared_ptr<Node>
            QuantizedConvolutionBiasAddBuilder(const Output& input,
                                               const Output& filters,
                                               const Output& bias,
                                               const Output& sum_input,
                                               const Strides& window_movement_strides,
                                               const Strides& window_dilation_strides,
                                               const CoordinateDiff& padding_below,
                                               const CoordinateDiff& padding_above,
                                               const Strides& data_dilation_strides,
                                               const Output& min_input,
                                               const Output& max_input,
                                               const Output& min_filter,
                                               const Output& max_filter,
                                               const Output& min_output,
                                               const Output& max_output,
                                               const Output& min_sum_input,
                                               const Output& max_sum_input,
                                               const bool with_relu = false);

        std::shared_ptr<Node>
            QuantizedConvolutionBiasSignedAddBuilder(const Output& input,
                                                     const Output& filters,
                                                     const Output& bias,
                                                     const Output& sum_input,
                                                     const Strides& window_movement_strides,
                                                     const Strides& window_dilation_strides,
                                                     const CoordinateDiff& padding_below,
                                                     const CoordinateDiff& padding_above,
                                                     const Strides& data_dilation_strides,
                                                     const Output& min_input,
                                                     const Output& max_input,
                                                     const Output& min_filter,
                                                     const Output& max_filter,
                                                     const Output& min_output,
                                                     const Output& max_output,
                                                     const Output& min_sum_input,
                                                     const Output& max_sum_input,
                                                     const bool with_relu = false);
    }
}
