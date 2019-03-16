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

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"
#include "ngraph/util.hpp"

#include <memory>

namespace ngraph
{
    namespace op
    {
        /// \brief Elementwise Relu operation.
        ///
        class Relu : public ngraph::op::util::UnaryElementwiseArithmetic
        {
        public:
            /// \brief Constructs a Relu operation.
            ///
            /// \param arg Node that produces the input tensor.
            Relu(const NodeOutput& arg);

            virtual std::shared_ptr<Node>
                copy_with_new_source_outputs(const OutputVector& new_args) const override;

            virtual void build_backprop(autodiff::Adjoints& adjoints,
                                        const OutputVector& deltas) override;
        };

        /// \brief Elementwise ReluBackprop operation.
        ///
        class ReluBackprop : public ngraph::op::util::BinaryElementwiseArithmetic
        {
        public:
            /// \brief Constructs a ReluBackprop operation.
            ///
            /// \param arg Output that produces the relu forward input tensor.
            /// \param delta Output that produces the backprop delta tensor.
            ReluBackprop(const NodeOutput& arg, const NodeOutput& delta);

            virtual std::shared_ptr<Node>
                copy_with_new_source_outputs(const OutputVector& new_args) const override;
        };
    }
}
