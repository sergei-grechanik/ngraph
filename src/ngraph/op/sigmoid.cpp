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

#include "ngraph/op/sigmoid.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

shared_ptr<Node>
    op::Sigmoid::copy_with_new_source_outputs(const OutputVector& new_source_outputs) const
{
    check_new_source_outputs_count(this, new_source_outputs);
    return make_shared<Sigmoid>(new_source_outputs.at(0));
}

op::Sigmoid::Sigmoid(const NodeOutput& arg)
    : UnaryElementwiseArithmetic("Sigmoid", arg)
{
    constructor_validate_and_infer_types();
}

op::SigmoidBackprop::SigmoidBackprop(const NodeOutput& arg, const NodeOutput& delta)
    : BinaryElementwiseArithmetic("SigmoidBackprop", arg, delta)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node>
    op::SigmoidBackprop::copy_with_new_source_outputs(const OutputVector& new_source_outputs) const
{
    check_new_source_outputs_count(this, new_source_outputs);
    return make_shared<SigmoidBackprop>(new_source_outputs.at(0), new_source_outputs.at(1));
}

void op::Sigmoid::build_backprop(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    auto delta = deltas.at(0);

    auto backprop = make_shared<op::SigmoidBackprop>(get_input_source_output(0), delta);

    adjoints.add_output_delta(get_input_source_output(0), backprop);
}
