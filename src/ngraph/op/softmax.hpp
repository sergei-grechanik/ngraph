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

#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"

namespace ngraph
{
    namespace op
    {
        // TODO(amprocte): This is not an elementwise operation, and should not subclass
        // UnaryElementwiseArithmetic.

        /// \brief Softmax operation.
        ///
        class Softmax : public util::UnaryElementwiseArithmetic
        {
        public:
            /// \brief Constructs a softmax operation.
            ///
            /// \param arg Output that produces the first input tensor.<br>
            /// `[d0, ...]`
            /// \param axes The axis positions (0-based) on which to calculate the softmax.
            ///
            /// Output `[d0, ...]`
            ///
            Softmax(const NodeOutput& arg, const AxisSet& axes);

            virtual std::shared_ptr<Node>
                copy_with_new_source_outputs(const OutputVector& new_source_outputs) const override;

            const AxisSet& get_axes() const { return m_axes; }
        protected:
            virtual void build_backprop(autodiff::Adjoints& adjoints,
                                        const OutputVector& deltas) override;

        private:
            AxisSet m_axes;
        };
    }
}
