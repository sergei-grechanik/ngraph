//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <memory>

#include "acosh.hpp"
#include "default_opset.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                OutputVector acosh(const Node& node)
                {
                    Output<ngraph::Node> data{node.get_ng_inputs().at(0)};

                    // Define inverse hyperbolic cosine in terms of natural logarithm:
                    //
                    // arccosh(x) = ln(x + sqrt(x^2 - 1))
                    //

                    Output<ngraph::Node> one_node{default_opset::Constant::create(
                        data->get_element_type(),
                        data->get_shape(),
                        std::vector<float>(ngraph::shape_size(data->get_shape()), 1.f))};

                    Output<ngraph::Node> sqrt_node{
                        std::make_shared<default_opset::Sqrt>(data * data - one_node)};

                    return {std::make_shared<default_opset::Log>(data + sqrt_node)};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
