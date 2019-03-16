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

#include "ngraph/op/atan.hpp"

#include "ngraph/axis_set.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/shape.hpp"

#include <string>
#include <vector>

using namespace std;
using namespace ngraph;

op::Atan::Atan(const NodeOutput& arg)
    : UnaryElementwiseArithmetic("Atan", arg)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node>
    op::Atan::copy_with_new_source_outputs(const OutputVector& new_source_outputs) const
{
    check_new_source_outputs_count(this, new_source_outputs);
    return make_shared<Atan>(new_source_outputs.at(0));
}

void op::Atan::build_backprop(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = get_input_source_output(0);

    auto one = make_shared<op::Constant>(x.get_element_type(), Shape{}, vector<string>{"1"});

    AxisSet axes;
    for (size_t i = 0; i < x.get_shape().size(); i++)
    {
        axes.insert(i);
    }
    auto ones = make_shared<op::Broadcast>(one, x.get_shape(), axes);

    adjoints.add_output_delta(x, delta / (ones + x * x));
}
