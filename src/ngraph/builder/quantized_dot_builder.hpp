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
#include "ngraph/op/experimental/quantized_dot_bias.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/quantized_dot.hpp"
#include "quantization_utils.hpp"

namespace ngraph
{
    namespace builder
    {
        std::shared_ptr<Node> QuantizedDotBuilder(const Output& input0,
                                                  const Output& input1,
                                                  const size_t reduction_axes_count,
                                                  const Output& min_input0,
                                                  const Output& max_input0,
                                                  const Output& min_input1,
                                                  const Output& max_input1,
                                                  const Output& min_output,
                                                  const Output& max_output,
                                                  const ngraph::element::Type& output_type,
                                                  const ngraph::AxisSet& input0_axes,
                                                  const ngraph::AxisSet& input1_axes,
                                                  const ngraph::AxisSet& output_axes);

        std::shared_ptr<Node> QuantizedDotBiasBuilder(const Output& input,
                                                      const Output& filters,
                                                      const Output& bias,
                                                      const Output& min_input,
                                                      const Output& max_input,
                                                      const Output& min_filter,
                                                      const Output& max_filter,
                                                      const Output& min_output,
                                                      const Output& max_output,
                                                      const bool requantize = true,
                                                      const bool with_relu = false);
    }
}
